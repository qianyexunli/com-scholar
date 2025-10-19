# retriever_local.py
import os, json, re, hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

# =============== 通用工具（和你现有逻辑对齐） ===============
def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _parse_ref_block(block: str) -> set:
    refs = set()
    parts = re.split(r"[,\s]+", block.strip())
    for p in parts:
        if "-" in p:
            try:
                a, b = map(int, p.split("-"))
                refs.update(range(a, b + 1))
            except:
                continue
        else:
            try:
                refs.add(int(p))
            except:
                continue
    return refs

def extract_references_from_chunk(doc: Document) -> Dict[str, str]:
    """从 chunk 文本里解析 [1,2-4] 这种标注并映射到 metadata['references']。"""
    text = doc.page_content
    references = doc.metadata.get("references", {}) or {}
    ref_blocks = re.findall(r"\[([\d,\-\s]+)\]", text)
    all_refs = set()
    for block in ref_blocks:
        all_refs.update(_parse_ref_block(block))
    all_refs = sorted(all_refs)
    return {str(r): references.get(str(r)) for r in all_refs}

def _md_split(text: str) -> List[str]:
    # 先按 Markdown 章节切，再对超长文本递归切分（与你现有策略一致）
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#",1),("##",2),("###",3)])
    parts = header_splitter.split_text(text)
    recu = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    out = []
    for p in parts:
        t = p.page_content
        if len(t) > 3000:
            out.extend(recu.split_text(t))
        else:
            out.append(t)
    return out

# =============== 构建与查询（本地 FAISS） ===============
class LocalJsonIndex:
    def __init__(self, embed_model: str = "BAAI/bge-m3"):
        # 多语模型，支持中英混合；如需更轻量可换 bge-small
        self.emb = HuggingFaceBgeEmbeddings(
            model_name=embed_model,
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vs: Optional[FAISS] = None
        self.docs: List[Document] = []
        self.id_by_hash: Dict[str, int] = {}

    def _doc_from_generic_json(self, item: dict) -> Optional[Document]:
        """
        适配通用 JSON：优先拼 'Title' + 'Abstract'，否则把字符串字段拼接成 text。
        “参考文献”如果是列表[{idx: ref}, ...] 会转成 dict 存到 metadata['references']。
        """
        # 文本候选
        title = item.get("Title") or item.get("title") or ""
        abstract = item.get("Abstract") or item.get("abstract") or item.get("text") or ""
        text = f"{title}\n\n{abstract}".strip()
        if not text:
            # fallback：拼接所有 str 字段
            str_vals = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
            text = "\n".join(str_vals).strip()
        if not text:
            return None

        # 引用字段适配：列表[{ "1": "..."}] -> dict
        refs = {}
        cand = item.get("references") or item.get("Refs") or item.get("Reference") or item.get("REFERENCE")
        if isinstance(cand, list):
            for r in cand:
                if isinstance(r, dict):
                    for k, v in r.items():
                        refs[str(k)] = v
        elif isinstance(cand, dict):
            refs = {str(k): v for k, v in cand.items()}

        meta = {"title": title or item.get("Source") or item.get("Source Title") or "Unknown", "references": refs}
        # 切分
        chunks = _md_split(text)
        docs = [Document(page_content=c, metadata=meta) for c in chunks if c.strip()]
        return docs

    def _iter_json_files(self, root: Path) -> Iterable[Path]:
        for p in root.rglob("*.json"):
            if p.is_file():
                yield p

    def build_from_dir(self, dir_path: str) -> None:
        root = Path(dir_path)
        if not root.exists():
            raise FileNotFoundError(f"本地目录不存在：{dir_path}")

        docs: List[Document] = []
        for fp in self._iter_json_files(root):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            # 支持：单对象 / 列表
            items = data if isinstance(data, list) else [data]
            for it in items:
                if not isinstance(it, dict):
                    continue
                made = self._doc_from_generic_json(it)
                if not made:
                    continue
                docs.extend(made)

        if not docs:
            raise RuntimeError("未从本地 JSON 中提取到文档内容。")

        # 建立 id 与 hash 索引，便于引用回填
        self.docs = docs
        self.id_by_hash = {text_hash(d.page_content): i for i, d in enumerate(self.docs)}
        self.vs = FAISS.from_texts([d.page_content for d in self.docs], self.emb, metadatas=[d.metadata for d in self.docs])

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.vs:
            raise RuntimeError("索引未初始化。")
        hits = self.vs.similarity_search_with_score(query, k=k)
        out = []
        for d, score in hits:
            idx = self.id_by_hash.get(text_hash(d.page_content))
            if idx is None:
                continue
            doc = self.docs[idx]
            refs = extract_references_from_chunk(doc)
            out.append({
                "chunk": d.page_content,
                "title": doc.metadata.get("title", "Unknown"),
                "references": {k: v for k, v in refs.items() if v},
                "score": float(score),
            })
        return out

# =============== 兼容你现有 retriever_weaviate 接口 ===============
_local_index: Optional[LocalJsonIndex] = None

def ensure_local_index(dir_path: str, embed_model: str = "BAAI/bge-m3") -> None:
    global _local_index
    if _local_index is None:
        _local_index = LocalJsonIndex(embed_model=embed_model)
        _local_index.build_from_dir(dir_path)

def retriever_local(question: str, dir_path: str, limit: int = 5) -> List[Dict]:
    ensure_local_index(dir_path)
    return _local_index.search(question, k=limit)
