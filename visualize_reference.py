import json
import re
import streamlit as st
from typing import List, Optional, Dict

def display_references(result):
    current_index = 1
    title_to_index = {}
    reference_list = []
    if result:
        all_results = result['result']
        all_references = result['reference']
        for num, part in enumerate(all_results):
            answer = part['answer']
            details = part['details']
            current_ref_ids = []
            if details:
                for detail in details:
                    title = detail.get("title")
                    if title and title not in title_to_index:
                        title_to_index[title] = current_index
                        reference_list.append({
                            "index": current_index,
                            "title": title,
                            "context": detail.get("context"),
                            "analyze": detail.get("analyze"),
                        })
                        current_index += 1
                    ref_id = title_to_index.get(title)
                    if ref_id is not None:
                        current_ref_ids.append(ref_id)

            ref_marks = ''.join([f'[{i}]' for i in sorted(set(current_ref_ids))])
            st.markdown(f"{answer} {ref_marks}", unsafe_allow_html=True)

            if details:
                with st.expander("🔍 References of this answer fragment"):
                    for detail in details:
                        ref_index = title_to_index.get(detail["title"])
                        st.markdown(f"Reference[{ref_index}]:", unsafe_allow_html=True)
                        st.markdown(detail["context"], unsafe_allow_html=True)
                        st.markdown(f"Source title: {detail['title']}")
                        if detail["analyze"]:
                            st.markdown(f"Analyze{detail['analyze']}")
                            
        with st.expander("📚 References List"):
            for ref in all_references:
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


def display_mixed(context_history: Dict[str, str]):
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
