import json
import re
import os
import hashlib
import asyncio
from tqdm.asyncio import tqdm
import weaviate
from weaviate.classes.init import Auth
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

all_documents = []

def split_text(md_text, title_for_metadata=""):
    # 用 Markdown 结构切分
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", 1), ("##", 2), ("###", 3)
    ])

    # 直接传入文本
    chunks = text_splitter.split_text(md_text)

    # 对较长 chunk 再做字符级别递归切分
    recu_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    final_chunks = []
    for chunk in chunks:
        text = chunk.page_content
        if len(text) > 3000:
            split_texts = recu_splitter.split_text(text)
            final_chunks.extend(split_texts)
        else:
            final_chunks.append(text)  # 注意：返回字符串，而非 Document

    return final_chunks 
    

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
    

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


def answer_generation(llm, question):
    template = '''
    You are a professional expert of optical frequency comb. Your task is to generate detailed answers to the given question.
    Use appropriate headings and subheadings (label '#'). But don't divide the answer into too many parts. 
    
    Question: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(question=question)
    response = llm.invoke(formatted_prompt)
    
    return response.content


def is_title(line):
    return re.match(r'^\s*#{1,6}\s+', line)


def is_non_empty_non_title(line):
    return line.strip() and not is_title(line)


def text_split(text):
    lines = text.splitlines()
    blocks = []
    buffer = []          
    title_buffer = []    

    for line in lines:
        if is_title(line):
            if buffer:
                blocks.append('\n'.join(buffer).strip())
                buffer = []
            title_buffer.append(line)
        elif is_non_empty_non_title(line):
            buffer.extend(title_buffer)
            title_buffer = []
            buffer.append(line)
        else:
            if title_buffer:
                title_buffer.append(line)
            else:
                buffer.append(line)
                
    if title_buffer:
        buffer.extend(title_buffer)
    if buffer:
        blocks.append('\n'.join(buffer).strip())

    return blocks


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


def final_retriever(llm, question, wcd_url, wcd_api_key, huggingface_key, limit=3):
    headers = {
        "X-HuggingFace-Api-Key": huggingface_key,
    }
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,                      
        auth_credentials=Auth.api_key(wcd_api_key),      
        headers=headers
    )

    client.connect()
    docs = client.collections.get("Sentence")


    text = answer_generation(llm, question)
    
    blocks = text_split(text)

    doc_list = []

    for i, doc in enumerate(blocks):
        retrieved_docs = []
        response = docs.query.bm25(
            query=doc,
            limit=limit,
        )
        for obj in response.objects:
            all_response = json.dumps(obj.properties, indent=2)
            all_response = json.loads(all_response)
            content = all_response.get("text", "")
            retrieved_docs.append(content)
            
        doc_list.append({
            "answer": doc,
            "docs": retrieved_docs,
        })

    client.close()

    return doc_list, text


def number_extraction(text):
    contexts_match = re.search(r"Numbers of the useful contexts:\s*(\[.*?\])", text)
    contexts = eval(contexts_match.group(1)) if contexts_match else []
  
    analyze_match = re.search(r"Analyze:\s*\n(.*)", text, re.DOTALL)
    analyze_block = analyze_match.group(1).strip() if analyze_match else ""

    analyze_items = []
    if analyze_block:
        pattern = r"- (Context \d+.*?)(?=\n- Context \d+|\Z)"  
        matches = re.findall(pattern, analyze_block, re.DOTALL)
        analyze_items = [re.sub(r'\s+', ' ', m).strip() for m in matches]

    if contexts_match:
        text = text.replace(contexts_match.group(0), '')
    if analyze_match:
        text = re.sub(r"Analyze:\s*\n.*", "", text, flags=re.DOTALL)

    cleaned_text = re.sub(r'\n{2,}', '\n\n', text).strip()

    return cleaned_text, contexts, analyze_items


def partial_answer_parallel(llm, question, wcd_url, wcd_api_key, huggingface_key):
    """
    并行生成带引用分析的优化答案，增加鲁棒性防止 NoneType 报错。
    """
    doc_list, text = final_retriever(llm, question, wcd_url, wcd_api_key, huggingface_key)

    template = '''
    You are a professional expert of optical frequency comb. 
    Your task is to make correction and supplement of the given answer fragment of the question with the help of the given contexts,
    give the serial number of the useful contexts, and give a detailed analyze that how these contexts contribute to the answer.
    You should first determine if the contexts are useful, then use this useful contexts to rewrite the answer.
    
    Pay attention to the subheadings in this answer fragment and make sure that the rewritten content is consistent with the title.
    The length of the rewritten answer is best to be a little longer than the original one.
    
    DO NOT mention phrases such as "the context", "in the picture", or references (like[1]) in your answer.
    MUST keep the headings and subheadings of each answer fragment.
    But DO NOT generate new headings and subheadings in the answer fragment.
    
    Give the numbers of the useful contexts and the analyze how these contexts contribute to the answer at the end of your answer and in the following format(DO NOT USE '#' to make a new subheading):
    
    Numbers of the useful contexts: [number1, number2, ...]
    
    Analyze: 
    - Context 1: ...

    - Context 2: ... 

-----------------------------------------------

    Question: {question}
    
    Answer Fragment: {answer}
    
    Contexts: {contexts}
    '''
    
    prompt = ChatPromptTemplate.from_template(template)

    def process_item(num, item):
        """
        单个小节处理函数（多线程调用）
        """
        answer = item['answer']
        contexts = item['docs']

        # 第一段原始生成的答案直接跳过修正
        if num == 0:
            return {
                "part": num + 1,
                "ori answer": answer,
                "answer": answer,
                "details": [],
            }

        # 调用模型生成增强答案
        formatted_prompt = prompt.format_messages(
            question=question, answer=answer, contexts=contexts
        )
        response = llm.invoke(formatted_prompt)
        new_text, numbers_list, analyze = number_extraction(response.content)

        # ---- 安全防护：numbers_list 为空或包含 None ----
        if not numbers_list:
            print(f"⚠️ No valid context numbers returned in part {num+1}.")
            numbers_list = []
        else:
            print(f"✅ Part {num+1} uses contexts:", numbers_list)

        # ---- 检查 analyze 是否匹配 ----
        if analyze and len(numbers_list) != len(analyze):
            print(f"⚠️ Warning: analyze length mismatch (numbers={len(numbers_list)}, analyze={len(analyze)})")

        context_analysis_dict = []
        for j, i in enumerate(numbers_list):
            if i is None or not isinstance(i, int):
                print(f"⚠️ Invalid index (None or not int): {i}")
                continue
            if i - 1 >= len(contexts) or i - 1 < 0:
                print(f"⚠️ Context index {i} out of range (len={len(contexts)})")
                continue

            analysis_text = re.sub(r'^Context \d+', '', analyze[j]) if j < len(analyze) else ""
            context_analysis_dict.append({
                "id": i,
                "context": contexts[i - 1],
                "analysis": analysis_text
            })

        part_result = []
        for item in context_analysis_dict:
            refs_collected = {}

            # ---- 修复 NoneType: doc_id 检查 ----
            doc_id = text_dict.get(text_hash(item['context']))
            if doc_id is None:
                print(f"⚠️ Warning: context not found in text_dict for hash={text_hash(item['context'])[:10]}...")
                continue

            try:
                chunk = all_documents[doc_id]
            except Exception as e:
                print(f"❌ Error accessing all_documents[{doc_id}]: {e}")
                continue

            # 提取引用
            refs = extract_references_from_chunk(chunk)
            refs_collected.update({k: v for k, v in refs.items() if v})

            part_result.append({
                'id': item['id'],
                "context": item['context'],
                "analyze": item['analysis'],
                "title": chunk.metadata.get("title", "Unknown"),
                "refs": refs_collected,
            })

        return {
            "part": num + 1,
            "ori answer": answer,
            "answer": new_text,
            "details": part_result,
        }

    # === 并行执行 ===
    from concurrent.futures import ThreadPoolExecutor, as_completed
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_item, num, item) for num, item in enumerate(doc_list)]
        results = [future.result() for future in as_completed(futures)]

    # === 排序 ===
    results.sort(key=lambda x: x["part"])

    # === 汇总结果 ===
    all_responses = "\n\n".join([r['answer'] for r in results if r.get('answer')])
    return results, all_responses, text



def answer_simplify(llm, question, ori_answer, answer):
    template = '''
    You are a professional expert of optical frequency comb.
    I will give you two versions of answer of the same given question.
    The second answer is a combination of the first answer and some relational literature contexts.
    Your task is to merge and optimize these two answers to form a more complete version.
    Don't give redundant and repetitive information in your answer.
    Do not mention phrases such as "in the context", "in the answer", or similar wording. 
    Use appropriate headings and subheadings(label '#').

    Question: {question}

    First answer: {ori_answer}

    Second answer: {answer}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(question=question, ori_answer=ori_answer, answer=answer)
    response = llm.invoke(formatted_prompt)
    return response.content