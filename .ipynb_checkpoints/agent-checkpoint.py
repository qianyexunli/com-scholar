import logging
import asyncio
from PIL import Image
import matplotlib.pyplot as plt
import json
from neo4j import GraphDatabase
from py2neo import Graph as Py2NeoGraph
from typing import List, Optional, Dict
from graph_chain import build_graph_qa_chain
from retriever_weaviate import retriever_weaviate
from answer_generation import answer_generation, chunk_output
from determine_database import final_answer, database_recall
from retriever_answer_parallel import partial_answer_parallel

uri = "neo4j+s://03e29abd.databases.neo4j.io"
username = "neo4j"
password = "50n7LTYnSOMbSuyfbrF2qA7LD8cSNmgFy6B3seFGklo"

file_path = "entity_cache.json"
with open(file_path, 'r', encoding='utf-8') as f:
    entity_cache = json.load(f)


max_steps=3
graph_db = "Literature Graph Database"
text_db = "Literature Text Database"


def handle_graph_db(strategy: str, llm=None) -> dict:
    graph = Py2NeoGraph(uri, auth=("neo4j", password))
    graph_qa_chain = build_graph_qa_chain(llm, graph, entity_cache)
    result = graph_qa_chain.invoke({
        "question": f"the entities needs to be searched{strategy}"
    })
    if not result:
        return {
            "chunks": [],
            "text": "No relevant papers are found in the graph database."
        }

    text = (
        f"cypher query: {result['cypher']} "
        f"context retrieved: {result['context']}"
    )

    result_dict = {
        "chunks": result,
        "text": text,
    }
    return result_dict


def handle_text_db(strategy: str, wcd_url=None, wcd_api_key=None, huggingface_key=None) -> dict:
    chunks = retriever_weaviate(strategy, wcd_url, wcd_api_key, huggingface_key)
    context_list = [{'chunk': item['chunk'], 'title': item['title']} for item in chunks]
    text = "relative chunks and source titles: " + json.dumps(context_list)
    result_dict = {
        "chunks": chunks,
        "text": text,
    }
    return result_dict


db_handles = {
    graph_db: handle_graph_db,
    text_db: handle_text_db
}


def fetch_context(
    database: str,
    strategy: str,
    llm=None,
    wcd_url=None,
    wcd_api_key=None,
    huggingface_key=None
) -> dict:
    try:
        handler = db_handles[database]
        if database == graph_db:
            return handler(strategy, llm)
        elif database == text_db:
            return handler(strategy, wcd_url, wcd_api_key, huggingface_key)
    except KeyError:
        raise ValueError(f"Unknown database type: {database}")


def run_mixed_type_query(
    llm, question: str, parsed_output: Dict, wcd_url, wcd_api_key, huggingface_key,
) -> dict:
    contexts: Dict[str, str] = {}

    current_db = parsed_output['first_database_to_call']
    current_strategy = parsed_output['call_strategy']
    current_method = parsed_output['methods']

    result_list = []
    result_dict = {}

    for step in range(1, max_steps + 1):
        context = fetch_context(
            current_db,
            current_strategy,
            llm=llm,
            wcd_url=wcd_url,
            wcd_api_key=wcd_api_key,
            huggingface_key=huggingface_key,
        )

        text = context["text"]
        contexts[f"context of step {step} from {current_db}"] = text
        print(contexts)

        if step == max_steps:
            logging.warning("Reached maximum steps. Skipping further parsing.")
            break

        parsed, response = database_recall(llm, question, contexts, parsed_output['call_strategy'])
        print(parsed)

        result_list.append({
            "step": step,
            "database": current_db,
            "strategy": current_strategy,
            "parsed": parsed,
            "text": text,
            "contexts": context["chunks"],
        })
    
        if parsed.get('sufficient', False):
            break

        current_db = parsed.get('next database')
        current_strategy = parsed.get('strategy')
        current_method = parsed.get('reason')

    final_response = final_answer(llm, question, contexts, parsed_output['call_strategy'])

    result_dict = {
        "parsed": parsed_output,
        "result": result_list,
        "response": final_response,
    }

    return result_dict


def agent(llm, question, parsed_output, wcd_url, wcd_api_key, huggingface_key):
    question_type = parsed_output['question_type']
    database = parsed_output['first_database_to_call']
    method = parsed_output['methods']
    strategy = parsed_output['call_strategy']

    if question_type == 'Entity-Type':
        try:
            graph = Py2NeoGraph(uri, auth=("neo4j", password))
            print(graph.run("RETURN 'Aura connected!'").data())
        except Exception as e:
            print("连接失败：", e)
        graph_qa_chain = build_graph_qa_chain(llm, graph, entity_cache)
        final_response = graph_qa_chain.invoke({"question": question})
        return final_response

    elif question_type == 'Knowledge-Type':
        all_result, final_response, text = partial_answer_parallel(llm, question, wcd_url, wcd_api_key, huggingface_key)
        dict = {
            "answer": final_response,
            "result": all_result,
            "text": text
        }
        return dict

    elif question_type == "Mixed-Type":
        answer = run_mixed_type_query(llm, question, parsed_output, wcd_url, wcd_api_key, huggingface_key)
        return answer


def answer_visualize(answer):
    final_answer = answer['answer']
    contexts = answer['contexts']
    result = answer['result']
    for i, item in enumerate(contexts):
        chunks = item[f"retrieved docs of part {i+1}"]
        grouped_data = {}
        for chunk in chunks:
            part = chunk['answer']
            context = chunk['context']
            title = chunk['title']
            refs = chunk['refs']
            
            if part not in grouped_data:
                grouped_data[part] = []
            grouped_data[part].append({
                "context": context,
                "title": title,
                "refs": refs,
            })
        
        # 输出分组后的结果
        for part, list in grouped_data.items():
            print(f"Original answer part: {part}")
            for single in list:
                print(f"Retrieved chunk: {single['context']}")
                
                title = single['title']
                refs = single['refs']
                print(f"source paper: {title}")
                if refs:
                    print("references:")
                    for k, v in refs.items():
                        print(f"[{k}] {v}")
                else:
                    print("No reference mentioned")  
                print("-" * 40 )

    ori_answer = answer['text']
    print(f"Original answer:{ori_answer}")
    print(f"Final answer:{final_answer}")


