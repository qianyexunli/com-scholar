import os
import asyncio
import streamlit as st
from langchain_community.chat_models import ChatTongyi
import json
from pathlib import Path
from datetime import datetime
from determine_database import determine_database
from memory import determine_database_with_history
from agent import agent

import streamlit.components.v1 as components
import networkx as nx
from visualize_graph import context_and_entities_to_graph, visualize_graph

FEEDBACK_FILE = "feedback_log.json"
MERGED_JSON_PATH = 'papers.json'  
PAPER_INFO_PATH = 'paper_info.json'
CHAT_FILE = 'chat_sessions.json'
HISTORY_FILE = Path("chat_history.json")

st.set_page_config(page_title="Agent", layout="wide")

# === 📌 Page Title ===
st.title("comb scholar")

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


paper_data = load_paper_data(MERGED_JSON_PATH)
file_names = load_file_names(paper_data)

paper_info_dict = load_paper_data(PAPER_INFO_PATH)


def normalize_title(title):
    return title.lower().replace(":", "").replace("?", "").replace("/", "").replace("_", "").replace("<", "").strip()


paper_info_dict_normalized = {
    normalize_title(title): info for title, info in paper_info_dict.items()
}

# === ✅ Remove timestamp from feedback logging ===
def save_feedback_only(feedback_type, comment):
    feedback_data = {
        # "timestamp": datetime.now().isoformat(),  # ⛔️ Removed per request
        "type": feedback_type,
        "comment": comment
    }
    os.makedirs("feedback", exist_ok=True)
    filename = os.path.join("feedback", "sidebar_feedback_log.json")
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
        existing.append(feedback_data)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.sidebar.error(f"❌ Feedback save failed: {e}")
        

if "api_key" not in st.session_state:
    st.session_state.api_key = "sk-58c2a9a30bf74bc0bd69688acc27c83e"
if "base_url" not in st.session_state:
    st.session_state.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
if "weaviate_url" not in st.session_state:
    st.session_state.weaviate_url = "https://y2nhd32gq4cby7u0zzlapg.c0.asia-southeast1.gcp.weaviate.cloud"
if "weaviate_key" not in st.session_state:
    st.session_state.weaviate_key = "YUhVcTR3bVh1Qk1rRVVxUF9sV2Fobi8vUnJnbG1wVkVPVE8wSkNIaGxzZEVic3FqM25ldWEyUG83eXl3PV92MjAw"
if "huggingface_key" not in st.session_state:
    st.session_state.huggingface_key = "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"
if "model" not in st.session_state:
    st.session_state.model = 'qwen-turbo-0919'
if "history" not in st.session_state:
    st.session_state.history = []  # 存储历史对话
if "parsed_output" not in st.session_state:
    st.session_state.parsed_output = None
if "result" not in st.session_state:
    st.session_state.result = {}
if "ori_answer" not in st.session_state:
    st.session_state.ori_answer = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions["Chat 1"] = []
    st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]


# ============ 🔧 设置区域（侧边栏） ============
with st.sidebar:
    with st.expander("🔧 Setting"):
        st.session_state.api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)
        st.session_state.base_url = st.text_input("Base URL", value=st.session_state.base_url)
        st.session_state.weaviate_url = st.text_input("Weaviate URL", value=st.session_state.weaviate_url)
        st.session_state.weaviate_key = st.text_input("Weaviate API Key", type="password", value=st.session_state.weaviate_key)
        st.session_state.huggingface_key = st.text_input("Huggingface API Key", value=st.session_state.huggingface_key)

        model_options = ['qwen-max-2025-01-25', 'qwen-max', 'qwen-max-latest', 'qwen-turbo', 'qwen-turbo-0919']  # 添加可选模型
        st.session_state.model = st.selectbox("Model Selection", model_options, index=model_options.index(st.session_state.model))

        show_debug = st.checkbox("Display Question Type Selection")
        show_ori = st.checkbox("Display Original Answer")
        show_answer = st.checkbox("Display Final Answer")
        show_context = st.checkbox("Display Reference", value=True)
        show_result = st.checkbox("Display Result List")

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
                    st.markdown(f"refs: [{ref_index}]")
                    st.markdown(ref_content)
                else:
                    st.warning(f"The reference numbered [{ref_index}] was not found")

    st.markdown("""
        <div style="background-color: #f0f8ff; padding: 15px; border-left: 5px solid #1890ff; border-radius: 5px; font-size: 16px;">
        <b>Tips：</b>Please distinguish the reference paragraphs of the answer and the references marked in the reference paragraphs<br>
        You can search the references marked in the reference paragraphs in the sidebar.
        </div>
        """, unsafe_allow_html=True)

chat_history = st.session_state.chat_sessions[st.session_state.current_session]

for turn in chat_history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])

question = st.chat_input("Ask something about kerr comb.")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    if not st.session_state.api_key or not st.session_state.base_url:
        st.error("Please input API Key and Base URL in the sidebar.")
    else:
        with st.spinner(f"dealing with `{st.session_state.model}`..."):
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key
            try:
                # 初始化模型
                llm = ChatTongyi(
                    model=st.session_state.model,
                    dashscope_api_key=st.session_state.api_key
                )

                weaviate_url = st.session_state.weaviate_url
                weaviate_api = st.session_state.weaviate_key
                huggingface_key = st.session_state.huggingface_key

                if not chat_history:
                    parsed_output, response = determine_database(question, llm)
                else:
                    parsed_output, response = determine_database_with_history(question, llm, chat_history)
                    
                result = agent(llm, question, parsed_output, weaviate_url, weaviate_api, huggingface_key)

                st.markdown("""
                            <script type="text/javascript"
                                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
                            </script>
                        """, unsafe_allow_html=True)

                question_type = parsed_output['question_type']

                if question_type == 'Knowledge-Type':
                    st.session_state.parsed_output = parsed_output
                    st.session_state.ori_answer = result['text']
                    st.session_state.answer = result['answer']
                    st.session_state.result = result['result']
                    references = result.get("references", result.get("reference", []))
                    
                    new_record = {
                        "user": question,
                        "assistant": result['answer'],
                    }
                    
                    if not any(record.get("question") == question for record in st.session_state.history):
                        st.session_state.history.append(new_record)

                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

                    if len(chat_history) == 0 and st.session_state.current_session.startswith("Chat "):
                        new_name = auto_rename_session(question)
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(st.session_state.current_session)
                        st.session_state.current_session = new_name

                    st.session_state.chat_sessions[st.session_state.current_session].append(new_record)

                    save_chat_sessions()
                    
                    with st.expander("📚 Full Reference List"):
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
                                            
        
                    if show_ori and st.session_state.ori_answer:
                        st.subheader("📊 Original Answer(Only for Text Database)")
                        st.text_area("", value=result['text'], height=400)

                    if show_answer and st.session_state.answer:
                        st.subheader("📊 Answer")
                        st.markdown(result['answer'])
          
                elif question_type == "Entity-Type":
                    entities = result["extract"]
                    corrected = result["entities"].dict()
                    cypher = result["cypher"]
                    retrieved = result['context']

                    st.markdown(result['answer'])

                    new_record = {
                        "user": question,
                        "assistant": result['answer'],
                    }
                    

                    if len(chat_history) == 0 and st.session_state.current_session.startswith("Chat "):
                        new_name = auto_rename_session(question)
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(st.session_state.current_session)
                        st.session_state.current_session = new_name

                    st.session_state.chat_sessions[st.session_state.current_session].append(new_record)

                    save_chat_sessions()
                    
                    if show_context:
                        with st.expander("📊 Retrieved Result"):
                            st.json(retrieved)

                    G = context_and_entities_to_graph(retrieved, corrected)

                    html_path = visualize_graph(G)

                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    components.html(html_content, height=600, scrolling=True)

                elif question_type == "Mixed-Type":
                    response = result["response"]
                    st.markdown(response, unsafe_allow_html=True)

                    new_record = {
                        "question": question,
                        "answer": response,
                    }
                    
                    if not any(record.get("question") == question for record in st.session_state.history):
                        st.session_state.history.append(new_record)

                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

                    if len(chat_history) == 0 and st.session_state.current_session.startswith("Chat "):
                        new_name = auto_rename_session(question)
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(st.session_state.current_session)
                        st.session_state.current_session = new_name

                    st.session_state.chat_sessions[st.session_state.current_session].append(new_record)

                    save_chat_sessions()

                    parsed = result["parsed"]
                    strategy = parsed["call_strategy"]
                    with st.expander("📚 Database Call Strategy"):
                        st.markdown(strategy)

                    result_list = result["result"]

                    if show_context:
                        for item in result_list:
                            db = item["database"]
                            step = item["step"]
                            st.subheader(f"Step {step}: Retrieved Results of {db}:")
                            if db == "Literature Text Database":
                                with st.expander("References of the answer:"):
                                    for part in item["contexts"]:
                                        st.markdown(part.get('chunk'), unsafe_allow_html=True)
                                        st.markdown(f"Source Title of the reference: {part.get('title')}")
                                    

                            elif db == "Literature Graph Database":
                                information = item['contexts']
                                if information:
                                    with st.expander("📊 Retrieved Result"):
                                        st.json(information['context'])
                                else:
                                    st.markdown("No Relative Papers Found")

                    st.session_state.result = result_list
    
                    if show_result and st.session_state.result:
                            with st.expander("Json Result"):
                                st.subheader("📊 Result List:")
                                st.json(result_list)
                                    
            except Exception as e:
                st.error(f"Error：{e}")

st.sidebar.markdown("### 📬 General Feedback")
feedback_type = st.sidebar.selectbox(
    "Feedback type",
    ["功能建议 Suggestion", "Bug 问题", "界面体验 UI", "其他 Other"],
    key="sidebar_feedback_type"
)

comment = st.sidebar.text_area("Your feedback", key="sidebar_feedback_text")

if st.sidebar.button("Submit", key="sidebar_feedback_submit"):
    if comment.strip():
        save_feedback_only(feedback_type, comment)
        st.sidebar.success("✅ Thank you for your feedback!")
    else:
        st.sidebar.warning("⚠️ Please enter some feedback text.")
