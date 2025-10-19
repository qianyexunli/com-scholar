import os
import streamlit as st
from langchain_openai import ChatOpenAI
import uuid
import json
import asyncio
import datetime
from supabase import create_client
from pathlib import Path
import streamlit.components.v1 as components
import networkx as nx
from typing import List, Optional, Dict

from visualize_reference import display_references, display_mixed
from langagent import build_agent, run_agent
from visualize_graph import context_and_entities_to_graph, visualize_graph

MERGED_JSON_PATH = 'papers.json'
PAPER_INFO_PATH = 'paper_info.json'
CHAT_FILE = 'chat_sessions.json'

st.set_page_config(page_title="μcombScholar", layout="wide")

@st.cache_data
def load_paper_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_file_names(data_dict):
    return list(data_dict.keys())

def save_chat_sessions():
    with open(CHAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.chat_sessions, f, indent=4, ensure_ascii=False)

def load_chat_sessions():
    path = CHAT_FILE
    if not os.path.exists(path):
        return {}  # 文件不存在，返回空字典

    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def auto_rename_session(question):
    name = question.strip().replace('\n', ' ')[:30]
    return f"{name}"

def normalize_title(title):
    return title.lower().replace(":", "").replace("?", "").replace("/", "").replace("_", "").replace("<", "").strip()


def init_supabase():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    return create_client(url, key)


def save_to_supabase(session_id, question, answer):
    sb = init_supabase()
    record = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    sb.table("chat_history").insert(record).execute()


# ============ 初始化 session_state ============
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("OPENAI_KEY", "")
if "base_url" not in st.session_state:
    st.session_state.base_url = st.secrets.get("BASE_URL", "")
if "weaviate_url" not in st.session_state:
    st.session_state.weaviate_url = st.secrets.get("WEAVIATE_URL", "")
if "weaviate_key" not in st.session_state:
    st.session_state.weaviate_key = st.secrets.get("WEAVIATE_KEY", "")
if "huggingface_key" not in st.session_state:
    st.session_state.huggingface_key = st.secrets.get("HUGGINGFACE_KEY", "")
if "uuid" not in st.session_state:
    st.session_state.uuid = str(uuid.uuid4())
if "model" not in st.session_state:
    st.session_state.model = "qwen-turbo-1101"
if "parsed_output" not in st.session_state:
    st.session_state.parsed_output = None
if "result" not in st.session_state:
    st.session_state.result = {}
if "answer" not in st.session_state:
    st.session_state.answer = None
if "ori_answer" not in st.session_state:
    st.session_state.ori_answer = None
if "paper_data" not in st.session_state:
    st.session_state.paper_data = load_paper_data(MERGED_JSON_PATH)
if "file_names" not in st.session_state:
    st.session_state.file_names = load_file_names(st.session_state.paper_data)
if "paper_info_dict_normalized" not in st.session_state:
    raw_info = load_paper_data(PAPER_INFO_PATH)
    st.session_state.paper_info_dict_normalized = {
        normalize_title(title): info for title, info in raw_info.items()
    }
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions["Chat 1"] = []
    st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]

wcd_url = st.session_state.weaviate_url
wcd_api_key = st.session_state.weaviate_key
huggingface_key = st.session_state.huggingface_key

paper_data = st.session_state.paper_data
file_names = st.session_state.file_names
paper_info_dict_normalized = st.session_state.paper_info_dict_normalized

with st.sidebar:
    st.markdown("## 🤖 **μcombScholar**")
    with st.expander("🔧 Setting", expanded=True):
        st.session_state.model = st.text_input("Model Selection", value=st.session_state.model)
        show_origin = st.checkbox("Display Origin Answer")
        show_answer = st.checkbox("Display Full Answer")

    with st.expander("💬Chat History"):
        if st.button("New Chat"):
            new_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_name] = []
            st.session_state.current_session = new_name
            save_chat_sessions()
        session_names = list(st.session_state.chat_sessions.keys())
        selected = st.radio("Select Session", session_names, index=session_names.index(st.session_state.current_session))
        st.session_state.current_session = selected
    
        # 手动重命名
        new_title = st.text_input("Rename Current", value=st.session_state.current_session)
        if new_title and new_title != st.session_state.current_session:
            st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(st.session_state.current_session)
            st.session_state.current_session = new_title
            save_chat_sessions()
    
        # 清除当前会话内容
        if st.button("Clear Current Chat"):
            st.session_state.chat_sessions[st.session_state.current_session] = []
            save_chat_sessions()

    with st.expander("🔍 Search refs", expanded=False):     
        selected_paper_prefix = st.text_input("Title", key="prefix_input")
        matched_names = []
        if selected_paper_prefix.strip():
            matched_names = [name for name in file_names if name.lower().startswith(selected_paper_prefix.lower())][:20]
        if matched_names:
            selected_paper = st.selectbox("Select Titles", matched_names, key="paper_select")
        else:
            selected_paper = None
            st.info("Please enter at least a few characters to match the name of the literature (up to the first 20 results )")
        
        ref_index = st.text_input("Num of refs（such as '1'）", key="ref_input")

        if st.button("Search", key="confirm_btn") and selected_paper:
            paper = paper_data[selected_paper]

            # --- 查找引用 ---
            if ref_index.strip():
                references = paper.get("references", [])
                ref_dict = {}
                for ref in references:
                    ref_dict.update(ref)
                ref_content = ref_dict.get(ref_index)
                if ref_content:
                    st.markdown(f"**📚 refs: [{ref_index}]**")
                    st.markdown(ref_content)
                else:
                    st.warning(f"The reference numbered [{ref_index}] was not found")

    st.markdown("""
        <div style="background-color: #f0f8ff; padding: 15px; border-left: 5px solid #1890ff; border-radius: 5px; font-size: 16px;">
        <b>Tips：</b>
        You can search the references marked in the reference paragraphs in the sidebar.
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
            <script type="text/javascript"
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
            </script>
        """, unsafe_allow_html=True)


chat_history = st.session_state.chat_sessions[st.session_state.current_session]

if chat_history:
    for turn in chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn["db_type"] == "text":
                display_references(turn['result'])
            elif turn['db_type'] == "graph":
                st.markdown(turn["answer"])
                with st.expander("📊 Retrieved Result"):
                    st.json(turn["references"])
            elif turn['db_type'] == "mixed":
                st.markdown(turn["answer"])
                result_list = turn['references']
                display_mixed(result_list)

question = st.chat_input("Ask something about microcomb.")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chatmessage("assistant"):
        with st.spinner("Thinking..."):
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key
            try:
                llm = ChatOpenAI(model=st.session_state.model, base_url=st.session_state.base_url)
                placeholder = st.empty() 
                graph = build_agent(wcd_url, wcd_api_key, huggingface_key)
                result = asyncio.run(run_agent(graph, llm, question, record=chat_history))
                question_type = result['question_type']
    
                if 'question_type' == 'Knowledge-Type':
                    st.session_state_ori_answer = result['ori_answer']
                    st.session_state.answer = result['answer']
                    st.session_state.result = result['retrieved_chunks']
    
                    references = result['references']
    
    
                    new_record = {
                        "question": question,
                        "answer": result['answer'],
                        "result": result,
                        "references": references,
                        "db_type": "text"
                    }
                    
                    if len(chat_history) == 0 and st.session_state.current_session.startswith("Chat "):
                        new_name = auto_rename_session(question)
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(st.session_state.current_session)
                        st.session_state.current_session = new_name
    
                    st.session_state.chat_sessions[st.session_state.current_session].append(new_record)
    
                    save_chat_sessions()
    
                    save_to_supabase(
                        session_id=st.session_state.uuid,
                        question=question,
                        answer=result["answer"],
                    )
    
                    with st.expander("📚 References List"):
                        for ref in references:
                            idx = ref['index']
                            raw_title = ref['title']
                            norm_title = normalize_title(raw_title)
                    
                            if norm_title in paper_info_dict_normalized:
                                paper_info = paper_info_dict_normalized[norm_title]
                                authors_str = ', '.join(paper_info['authors'])
                                source = paper_info['source']
                                year = paper_info['year']
                                st.markdown(f"[{idx}] {authors_str}, {raw_title}, {source}, {year}")
                            else:
                                st.markdown(f"[{idx}] {raw_title}")
                                st.markdown(f"- Context: {ref['context']}", unsafe_allow_html=True)
                                if ref['analyze']:
                                    st.markdown(f"- Analyze: {ref['analyze']}")  
                                    
                    if show_origin and st.session_state.ori_answer:
                        with st.expander('Original Answer:'):
                            st.markdown(st.session_state.ori_answer)
                                            
                    if show_answer and st.session_state.answer:
                        with st.expander('Full Answer:'):
                            st.markdown(st.session_state.answer)
    
                elif question_type == "Entity-Type":
                    result_entity = result['answer']
                    corrected = result_entity['entities'].dict()
                    retrieved = result_entity['context']
                    answer = result_entity['answer']
                    st.markdown(answer)
    
                    with st.expander("📊 Retrieved Result"):
                            st.json(retrieved)
    
                    G = context_and_entities_to_graph(retrieved, corrected)
    
                    html_path = visualize_graph(G)
    
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    components.html(html_content, height=600, scrolling=True)
    
                    new_record = {
                        "question": question,
                        "answer": answer,
                        "result": {},
                        "references": retrieved,
                        "db_type": "graph",
                    }
                    
    
                    if len(chat_history) == 0 and st.session_state.current_session.startswith("Chat "):
                        new_name = auto_rename_session(question)
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(st.session_state.current_session)
                        st.session_state.current_session = new_name
    
                    st.session_state.chat_sessions[st.session_state.current_session].append(new_record)
    
                    save_chat_sessions()
    
                    save_to_supabase(
                        session_id=st.session_state.uuid,
                        question=question,
                        answer=new_record['answer'],
                    )
    
                elif question_type == "Mixed-Type":
                    answer = result['answer']
                    contexts = result['context_history']
    
                    st.markdown(answer, unsafe_allow_html=True)
    
                    new_record = {
                            "question": question,
                            "answer": response,
                            "result": {},
                            "references": contexts,
                            "db_type": "mixed",
                        }
    
                
                    if len(chat_history) == 0 and st.session_state.current_session.startswith("Chat "):
                        new_name = auto_rename_session(question)
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(st.session_state.current_session)
                        st.session_state.current_session = new_name
    
                    st.session_state.chat_sessions[st.session_state.current_session].append(new_record)
    
                    save_chat_sessions()
    
                    save_to_supabase(
                        session_id=st.session_state.uuid,
                        question=question,
                        answer=response,
                    )
    
                    with st.expander('References:'):
                        for step_key in sorted(context_history.keys(), key=lambda x: int(x.split()[1])):
                            context_str = context_history[step_key]
                            db_type = "Literature Graph Database" if "graph" in step_key else "Literature Text Database"
                    
                            st.subheader(f"🔍 {step_key} ({db_type})")
                    
                            if db_type == "Literature Text Database":
                                try:
                                    context = json.loads(context_str)
                                    with st.expander("📚 References"):
                                        for ref in context:
                                            st.markdown(ref.get("chunk", ""), unsafe_allow_html=True)
                                            st.markdown(f"**Source**: {ref.get('title', '')}")
                                except Exception as e:
                                    st.warning(f"Failed to load text context: {e}")
                            
                            elif db_type == "Literature Graph Database":
                                try:
                                    context = eval(context_str) if isinstance(context_str, str) else context_str
                                    with st.expander("📊 Retrieved Result from Graph"):
                                        st.json(context)
                                except Exception as e:
                                    st.warning(f"Failed to load graph context: {e}")
    
            except Exception as e:
                st.error(f"Error：{e}")