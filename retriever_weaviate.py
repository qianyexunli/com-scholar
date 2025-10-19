import os
import json
import re
import hashlib
from typing import Set, List, Dict
from PIL import Image
import matplotlib.pyplot as plt
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
import weaviate
from weaviate.classes.init import Auth


# ========================
# 全局文档加载
# ========================
all_documents = []


def split_text(md_text, title_for_metadata=""):
    """将 Markdown 文本分块"""
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", 1), ("##", 2), ("###", 3)
    ])

    chunks = text_splitter.split_text(md_text)

    recu_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    final_chunks = []
    for chunk in chunks:
        text = chunk.page_content
        if len(text) > 3000:
            split_texts = recu_splitter.split_text(text)
            final_chunks.extend(split_texts)
        else:
            final_chunks.append(text)
    return final_chunks  # 返回字符串列表


def text_hash(text):
    """计算文本的 SHA256 哈希"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ========================
# 读取本地 papers.json 文件
# ========================
with open("papers.json", "r", encoding="utf-8") as f:
    merged_data = json.load(f)

all_documents = []

for real_title, data in merged_data.items():
    text = data.get("text", "")
    references = data.get("references", [])
    figs = data.get("figs", [])

    # 将 references 列表转换为字典
    ref_dict = {str(k): v for ref in references for k, v in ref.items()}

    chunks = split_text(text, real_title)
    for chunk in chunks:
        all_documents.append(Document(
            page_content=chunk,
            metadata={
                "title": real_title,
                "references": ref_dict,
                "figs": figs
            }
        ))

# 构建哈希索引表
text_data = []
text_dict = {}
for doc_id, chunk in enumerate(all_documents):
    text = chunk.page_content
    text_data.append({"text": text, "id": doc_id})
    key = text_hash(text)
    text_dict[key] = doc_id


# ========================
# 引用提取辅助函数
# ========================
def parse_ref_block(block: str) -> set:
    """解析引用块 [1,2-4] → {1,2,3,4}"""
    refs = set()
    parts = re.split(r"[,\s]+", block.strip())
    for part in parts:
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                refs.update(range(start, end + 1))
            except:
                continue
        else:
            try:
                refs.add(int(part))
            except:
                continue
    return refs


def extract_references_from_chunk(doc: Document) -> Dict[str, str]:
    """提取文中引用与 metadata 中的参考文献对应"""
    text = doc.page_content
    references = doc.metadata.get("references", {})
    ref_blocks = re.findall(r"\[([\d,\-\s]+)\]", text)

    all_refs = set()
    for block in ref_blocks:
        all_refs.update(parse_ref_block(block))
    all_refs = sorted(all_refs)
    matched_refs = {str(ref): references.get(str(ref)) for ref in all_refs}
    return matched_refs


# ========================
# Weaviate 检索函数
# ========================
def retriever_weaviate(question, wcd_url, wcd_api_key, huggingface_key, limit=5):
    """从 Weaviate 云端数据库检索相关文本块"""
    headers = {"X-HuggingFace-Api-Key": huggingface_key}

    # 连接 Weaviate 云实例
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=Auth.api_key(wcd_api_key),
        headers=headers
    )

    database = client.collections.get("Sentence")

    # 混合检索（文本 + 向量）
    response = database.query.bm25(query=question, limit=limit)

    matched_info = []
    for obj in response.objects:
        refs_collected = {}

        all_response = json.dumps(obj.properties, indent=2)
        all_response = json.loads(all_response)
        content = all_response.get("text", "")

        # 匹配本地缓存中的原始 chunk
        doc_id = text_dict.get(text_hash(content))
        if doc_id is not None:
            chunk = all_documents[doc_id]
            refs = extract_references_from_chunk(chunk)
            refs_collected.update({k: v for k, v in refs.items() if v})
            title = chunk.metadata["title"]
        else:
            title = all_response.get("title", "Unknown")

        matched_info.append({
            "chunk": content,
            "title": title,
            "references": refs_collected,
        })

    client.close()
    return matched_info
