import os
import json
import re
from typing import Set, List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.prompts import ChatPromptTemplate
import hashlib
from typing import List, Dict

import weaviate

from weaviate.classes.init import Auth

all_documents = []

def split_text(md_text, title_for_metadata=""):
    # 用 Markdown 结构切分
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", 1), ("##", 2), ("###", 3)
    ])

    # 直接传入文本
    chunks = text_splitter.split_text(md_text)

    # 对较长 chunk 再做字符级别递归切分
    recu_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
    final_chunks = []
    for chunk in chunks:
        text = chunk.page_content
        if len(text) > 2500:
            split_texts = recu_splitter.split_text(text)
            final_chunks.extend(split_texts)
        else:
            final_chunks.append(text)  # 注意：返回字符串，而非 Document

    return final_chunks 
    

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
def merge_json_files(folder_path):
    merged_data = {}

    for filename in os.listdir(folder_path):

        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    merged_data.update(data)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return merged_data

folder_path = "papers" 
merged_data = merge_json_files(folder_path)
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


text_data = []
text_dict = {}

for doc_id, chunk in enumerate(all_documents):
    text = chunk.page_content
    id = doc_id
    text_data.append({
        "text": text,
        "id": id,
    })
    key = text_hash(text)
    text_dict[key] = id


def parse_ref_block(block: str) -> set:
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
    text = doc.page_content
    references = doc.metadata.get("references", {})
    ref_blocks = re.findall(r"\[([\d,\-\s]+)\]", text)

    all_refs = set()
    for block in ref_blocks:
        all_refs.update(parse_ref_block(block))

    all_refs = sorted(all_refs)
    matched_refs = {str(ref): references.get(str(ref)) for ref in all_refs}
    return matched_refs


def retriever_weaviate(question, wcd_url, wcd_api_key, huggingface_key, limit=10):
    headers = {
        "X-HuggingFace-Api-Key": huggingface_key,
    }
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,                       # `weaviate_url`: your Weaviate URL
        auth_credentials=Auth.api_key(wcd_api_key),      # `weaviate_key`: your Weaviate API key
        headers=headers
    )

    database = client.collections.get("Chunk")
    
    response = database.query.hybrid(
        query=question,
        limit=limit,
    )

    matched_info = []
    
    for obj in response.objects:
        refs_collected = {}
        all_response = json.dumps(obj.properties, indent=2)
        all_response = json.loads(all_response)
        content = all_response.get("text", "")
        doc_id = text_dict.get(text_hash(content))
        chunk = all_documents[doc_id]
        refs = extract_references_from_chunk(chunk)
        refs_collected.update({k: v for k, v in refs.items() if v})
        
        matched_info.append({
            "chunk": content,
            "title": chunk.metadata['title'],
            "references": refs_collected,
        })

    client.close()

    return matched_info
        

    
        
