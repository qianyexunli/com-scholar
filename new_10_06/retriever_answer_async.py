import json
import re
import os
import ast
import hashlib
import weaviate
from tqdm.asyncio import tqdm_asyncio
import asyncio
from weaviate.classes.init import Auth
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import time
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    return SentenceTransformer("JALLAJ/5epo")


embed_model = load_model()


def split_text(md_text, file_path):

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", 1), ("##", 2), ("###", 3)
    ])

    chunks = text_splitter.split_text(md_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
    final_chunks = []
    for chunk in chunks:
        text = chunk.page_content
        if len(text) > 2500:
            final_chunks.extend(splitter.split_text(text))
        else:
            final_chunks.append(chunk.page_content)

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
    Generate a introduction first, than answer the question point by point.
    You MUST use appropriate headings and subheadings (label '#' and serial numbers) to better clarify this question. 
    
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


def final_retriever(llm, question, wcd_url, wcd_api_key, huggingface_key, limit=7):

    headers = {
        "X-HuggingFace-Api-Key": huggingface_key,
    }
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,                      
        auth_credentials=Auth.api_key(wcd_api_key),      
        headers=headers
    )

    client.connect()
    docs = client.collections.get("Chunk2")

    text = answer_generation(llm, question)
    
    blocks = text_split(text)

    doc_list = []

    for i, doc in enumerate(blocks):
        retrieved_docs = []
        query_vec = embed_model.encode(doc, normalize_embeddings=True).tolist()
        response = docs.query.hybrid(
            query=doc,
            vector=query_vec,
            alpha=0.3,
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

def number_documents(doc_list):
    return {i + 1: doc for i, doc in enumerate(doc_list)}


def extract_indices_from_output(output_text):  
    try:
        start = output_text.find('[')
        end = output_text.find(']', start) + 1
        list_str = output_text[start:end]
        return ast.literal_eval(list_str)
    except Exception as e:
        print("Failed to parse indices:", e)
        return []


def rerank(llm, doc):
    answer = doc['answer']
    contexts = number_documents(doc['docs'])
    
    template = '''
You are a professional expert in optical frequency combs.

You are given a paragraph that is part of a technical answer with a heading or subheading. 

Your task is to identify which documents are most relevant to supporting or enriching the heading and the content of this paragraph.

You will also be provided with a list of candidate documents. Each document has a number indicating its position in the list.

Select the **top 3 documents** that are most relevant to the given paragraph, considering their technical alignment, contextual consistency, and potential to enhance or validate the claims made.

Only output the indices (starting from 1) of the most relevant documents in a **Python-style list**, such as: [2, 4, 5].

Do not explain your choices.
    
Paragraph: {answer}

Contexts: {contexts}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(answer=answer, contexts=contexts)
    response = llm.invoke(formatted_prompt)

    nums = extract_indices_from_output(response.content)
    docs = [contexts[i] for i in nums if i in contexts]
    
    return docs, nums


def number_extraction(text):
    # === 提取 Numbers of the useful contexts（增强冒号容错） ===
    contexts_match = re.search(
        r"\*{0,2}Numbers of the useful contexts\*{0,2}\s*[:：]?\s*(\[[^\]]*\])",
        text,
        re.IGNORECASE
    )
    try:
        contexts = ast.literal_eval(contexts_match.group(1)) if contexts_match else []
    except Exception as e:
        print(f"[number_extraction] context list parse error: {e}")
        contexts = []

    # === 提取 Analyze 块（支持加粗、空格、大小写、冒号容错）===
    analyze_match = re.search(
        r"\*{0,2}Analyze\*{0,2}\s*[:：]\s*\n((?:.|\n)*?)\s*$",
        text.strip(),
        re.IGNORECASE | re.DOTALL
    )
    analyze_block = analyze_match.group(1).strip() if analyze_match else ""

    # === 提取每条 Context 分析项（增强格式兼容性）===
    analyze_items = []
    if analyze_block:
        pattern = (
            r"(?:-?\s*)?"                 # 可选的 `- ` 开头
            r"(?:\*\*)?"                  # 可选的 `**`
            r"Context\s+(\d+)"            # 匹配 `Context 1`
            r"(?:\*\*)?"                  # 可选的 `**`
            r"\s*[:：]?\s*"               # 冒号可选，前后有空格
            r"(.*?)(?="                   # 匹配内容直到下一个 Context 或结尾
            r"(?:\n(?:-?\s*)?(?:\*\*)?Context\s+\d+(?:\*\*)?\s*[:：]?)|\Z)"
        )
        matches = re.findall(pattern, analyze_block, re.IGNORECASE | re.DOTALL)
        clean_content = lambda c: re.sub(r'^(\*+)', '', c.strip())
        analyze_items = [f"Context {num}: {clean_content(content)}" for num, content in matches]

    # === 清除原始块 ===
    if contexts_match:
        text = text.replace(contexts_match.group(0), '')
    if analyze_match:
        text = text.replace(analyze_match.group(0), '')

    cleaned_text = re.sub(r'\n{2,}', '\n\n', text).strip()

    return cleaned_text, contexts, analyze_items



async def partial_answer_async(llm, question, wcd_url, wcd_api_key, huggingface_key):
    doc_list, text = final_retriever(llm, question, wcd_url, wcd_api_key, huggingface_key)
    containers = [st.container() for _ in range(len(doc_list))]
    all_results = [None] * len(doc_list)
    title_to_index = {}
    reference_list = []
    semaphore = asyncio.Semaphore(2)
    done_event = asyncio.Event()

    template = '''
    You are a professional expert of optical frequency comb. 
    Your task is to make correction and supplement of the given answer fragment of the question with the help of the given contexts,
    give the serial number of the useful contexts, and give a detailed analyze that how these contexts contribute to the answer.
    You should first determine if the contexts are useful, then use this useful contexts to rewrite the answer.
    But DO NOT output the original text and references(like[1]) from this contexts directly.
    
    Pay attention to the subheadings in this answer fragment and make sure that the rewritten content is consistent with the title.
    The length of the rewritten answer is best to be a little longer than the original one.
    
    DO NOT mention phrases such as "the context", "in the picture", or references in the given context.
    MUST keep the headings and subheadings of each answer fragment.  But DO NOT generate new headings or subheadings.
    ⚠️ Any new heading beginning with '#' (except the origin heading) will be considered invalid output and discarded. 
    
    Give the numbers of the useful contexts and the analyze how these contexts contribute to the answer at the end of your answer and in the following format:
    
    Numbers of the useful contexts : [number1, number2, ...] (Please limit the sequence number range of the context to [1, 2, 3] based on the order)
    
    Analyze: 
    - Context 1: ...

    - Context 2: ... 

-----------------------------------------------

    Question: {question}
    
    Answer Fragment: {answer}
    
    Contexts: {contexts}
    '''
    
    prompt = ChatPromptTemplate.from_template(template)

    async def process_item(num, item):
        try:
            answer = item['answer']
            
            if num == 0:
                return {
                    "part": num + 1,
                    "ori answer": answer,
                    "answer": answer,
                    "details": [],
                }

            contexts, nums = rerank(llm, item)
        
            formatted_prompt = prompt.format_messages(
                question=question, answer=answer, contexts=contexts
            )
            response = await llm.ainvoke(formatted_prompt)
            new_text, numbers_list, analyze = number_extraction(response.content)

            analyze_dict = {
                'part': num + 1,
                'ori' : answer,
                'new': new_text,
                'numbers': numbers_list,
                'analyze': analyze,
                'context': contexts,
                'response': response.content,
            }
        
                
            if analyze:
                if len(numbers_list) != len(analyze):
                    print("Numbers of analyze error!")
                context_analysis_dict = [
                    {
                        "id": i,
                        "context": contexts[i - 1],
                        "analysis": re.sub(r'^Context \d+', '', analyze[j])}
                    for j, i in enumerate(numbers_list)
                ]
            else:
                context_analysis_dict = [
                    {
                        "id": i,
                        "context": contexts[i - 1],
                        "analysis": '',
                    }
                    for j, i in enumerate(numbers_list)
                ]

            part_result = []
            for item in context_analysis_dict:
                refs_collected = {}
                doc_id = text_dict.get(text_hash(item['context']))
                if not doc_id:
                    st.markdown('No doc id!')
                    
                chunk = all_documents[doc_id]

                if chunk.page_content != item['context']:
                    st.markdown("Not the same!")

                part_result.append({
                    'id': item['id'],
                    "context": item['context'],
                    "analyze": item['analysis'],
                    "title": chunk.metadata["title"],
                    "refs": refs_collected,
                })

            return {
                "part": num + 1,
                "ori answer": answer,
                "answer": new_text,
                "details": part_result,
            }
            
        except Exception as e:
            st.markdown(f"Error in part {num}: {e}")
            return {
                "part": num + 1,
                "ori answer": "",
                "answer": "",
                "details": [],
            }

    async def handle_and_display(num, item):
        async with semaphore:
            part_result = await process_item(num, item)
            all_results[num] = part_result

    async def monitor_display():
        current_part = 0
        while current_part < len(doc_list):
            if all_results[current_part] is not None:
                part_result = all_results[current_part]
                answer = part_result['answer']
                details = part_result.get('details', [])
                
                with containers[current_part]:
                    if current_part == 0:
                        st.markdown(part_result['ori answer'])
                    else:
                        current_ref_ids = []
                        for detail in details:
                            title = detail.get("title")
                            if title not in title_to_index:
                                title_to_index[title] = len(reference_list) + 1
                                reference_list.append({
                                    "index": title_to_index[title],
                                    "title": title,
                                    "context": detail.get("context"),
                                    "analyze": detail.get("analyze"),
                                })
                            current_ref_ids.append(title_to_index[title])
                        
                        ref_marks = ''.join([f'[{i}]' for i in sorted(set(current_ref_ids))])
                        st.markdown(f"{answer} {ref_marks}", unsafe_allow_html=True)
                        
                        if details:
                            with st.expander("🔍 References of the answer"):
                                for detail in details:
                                    ref_index = title_to_index[detail["title"]]
                                    st.markdown(f"Reference[{ref_index}]:")
                                    st.markdown(detail["context"], unsafe_allow_html=True)
                                    st.markdown(f"Source title: {detail['title']}")
                                    if detail["analyze"]:
                                        st.markdown(f"Analyze: {detail['analyze']}")
                current_part += 1
            else:
                await asyncio.sleep(1)
        done_event.set()
            
    monitor_task = asyncio.create_task(monitor_display())
    tasks = [handle_and_display(num, item) for num, item in enumerate(doc_list)]
    await asyncio.gather(*tasks)
    await done_event.wait()

    all_responses = "\n\n".join([res['answer'] for res in all_results])
    return all_results, all_responses, text, reference_list
