from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from py2neo import Graph as Py2NeoGraph
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict
import json

from graph_chain import build_graph_qa_chain
from retriever_weaviate import retriever_weaviate
from determine_database import final_answer
from retriever_answer_async import partial_answer_async

from prompts import (
    determine_prompt,
    determine_history_prompt,
    recall_prompt,
)

uri = "neo4j+s://a36f6471.databases.neo4j.io"
username = "neo4j"
password = "E7QyiOiiDAi0SjQY7eXvUyjNPNzEHa4sYsMdlTqc1gI"

file_path = "entity_cache.json"
with open(file_path, 'r', encoding='utf-8') as f:
    entity_cache = json.load(f)

class AgentState(TypedDict):
    question: str
    question_type: Optional[str]
    parsed_output: Dict
    step: int
    database: Optional[str]
    strategy: Optional[str]
    context_history: Dict[str, str]
    retrieved_chunks: List[Dict]
    references: List[Any]
    sufficient: bool
    llm: object
    ori_answer: Optional[str]
    answer: Optional[str]
    record: List[Dict]
    retried_with_mixed: bool  
    original_question: str 

class QuestionAnalysis(BaseModel):
    question_type: Literal["Knowledge-Type", "Entity-Type", "Mixed-Type"]
    database_to_call: Literal["Literature Text Database", "Literature Graph Database", "Both"]
    first_database_to_call: str
    methods: Optional[str] = None
    call_strategy: str

class RecallDecision(BaseModel):
    sufficient: bool
    next_database: Optional[str] = None
    reason: Optional[str] = None
    strategy: Optional[str] = None


def build_agent(wcd_url, wcd_api_key, huggingface_key):
    def entry_node(state: AgentState) -> AgentState:
        return state
    
    def determine_question_type_node(state: AgentState) -> AgentState:
        prompt_template = ChatPromptTemplate.from_template(determine_prompt)
        model = state['llm'].with_structured_output(QuestionAnalysis)
        result = model.invoke(prompt_template.format_messages(question=state["question"]))
        print(result)
        state["question_type"] = result.question_type
        state["parsed_output"] = result.dict()
        state["database"] = result.first_database_to_call if result.question_type == "Mixed-Type" else result.database_to_call
        state["strategy"] = result.call_strategy
        return state

    def determine_question_with_history_node(state: AgentState) -> AgentState:
        history = state.get("record", [])
        new_records = [{"user": item["question"], "assistant": item["answer"]} for item in history]
        prompt_template = ChatPromptTemplate.from_template(determine_history_question)
        model = state['llm'].with_structured_output(QuestionAnalysis)
        result = model.invoke(prompt_template.format_messages(question=state['question'], record=new_records))
        state["question_type"] = result.question_type
        state["parsed_output"] = result.dict()
        state["database"] = result.first_database_to_call if result.question_type == "Mixed-Type" else result.database_to_call
        state["strategy"] = result.call_strategy
        return state

    def handle_entity_type(state: AgentState) -> AgentState:
        graph = Py2NeoGraph(uri, auth=("neo4j", password))
        graph_qa_chain = build_graph_qa_chain(state['llm'], graph, entity_cache)
        result = graph_qa_chain.invoke({"question": state["question"]})
        state["answer"] = result
        return state

    def check_entity_result_node(state: AgentState) -> AgentState:
        if not state["answer"] and not state["retried_with_mixed"]:
            state["retried_with_mixed"] = True
            state["question"] = state["original_question"] + " (Mixed-Type)"
        return state
        
    async def handle_knowledge_type(state: AgentState) -> AgentState:
        all_result, final_response, ori_answer, reference_list = await partial_answer_async(
            state["llm"], state["question"], wcd_url, wcd_api_key, huggingface_key
        )
        state["ori_answer"] = ori_answer
        state["answer"] = final_response
        state["retrieved_chunks"] = all_result
        state["references"] = reference_list
        return state
       
    def graph_retrieval_node(state: AgentState) -> AgentState:
        graph = Py2NeoGraph(uri, auth=("neo4j", password))
        graph_qa_chain = build_graph_qa_chain(state['llm'], graph, entity_cache)
        result = graph_qa_chain.invoke({"question": f"the entities needs to be searched{state['strategy']}"})
        state["context_history"][f"step {state['step']} from graph"] = str(result)
        state["retrieved_chunks"] = result
        state["step"] += 1
        return state
    
    def text_retrieval_node(state: AgentState) -> AgentState:
        chunks = retriever_weaviate(state['strategy'], wcd_url, wcd_api_key, huggingface_key, limit=4)
        context_list = [{'chunk': item['chunk'], 'title': item['title']} for item in chunks]
        state["context_history"][f"step {state['step']} from text"] = json.dumps(context_list)
        state["retrieved_chunks"] = chunks
        state["step"] += 1
        return state
       
    def recall_decision_node(state: AgentState) -> AgentState:
        prompt_template = ChatPromptTemplate.from_template(recall_prompt)
        model = state['llm'].with_structured_output(RecallDecision)
        context_str = json.dumps(state["context_history"], indent=2)
        result = model.invoke(
            prompt_template.format_messages(
                question=state["question"],
                context=context_str,
                strategy=state["strategy"]
            )
        )
        print(result)
        state["sufficient"] = result.sufficient
        if not result.sufficient:
            state["database"] = result.next_database
            state["strategy"] = result.strategy
        return state
       
    def final_answer_node(state: AgentState) -> AgentState:
        state["answer"] = final_answer(
            state["llm"], state["question"], state["context_history"], state["strategy"], state.get("record", "")
        )
        return state
     
    def route_by_question_type(state: AgentState) -> str:
        if state["question_type"] == "Entity-Type":
            return "handle_entity_type"
        elif state["question_type"] == "Knowledge-Type":
            return "handle_knowledge_type"
        else:
            return "graph_retrieval" if state["database"] == "Literature Graph Database" else "text_retrieval"
      
    def route_after_recall(state: AgentState) -> str:
        if state["sufficient"] or state['step'] >= 4:
            return "final_answer"
        else:
            return "graph_retrieval" if state["database"] == "Literature Graph Database" else "text_retrieval"

    def route_after_entity_check(state: AgentState) -> str:
        if state["retried_with_mixed"]:
            return "determine_question_type"
        else:
            return END
 
    def entry_router(state: AgentState) -> str:
        return "determine_with_history" if state.get("record") else "determine_question_type"

    builder = StateGraph(AgentState)
    builder.add_node("entry", entry_node)
    builder.add_node("determine_question_type", determine_question_type_node)
    builder.add_node("determine_with_history", determine_question_with_history_node)
    builder.add_node("handle_entity_type", handle_entity_type)
    builder.add_node("check_entity_result", check_entity_result_node)
    builder.add_node("handle_knowledge_type", handle_knowledge_type)
    builder.add_node("graph_retrieval", graph_retrieval_node)
    builder.add_node("text_retrieval", text_retrieval_node)
    builder.add_node("recall_decision", recall_decision_node)
    builder.add_node("final_answer", final_answer_node)

    builder.set_entry_point("entry")
    builder.add_conditional_edges("entry", entry_router)
    builder.add_conditional_edges("check_entity_result", route_after_entity_check)
    builder.add_conditional_edges("determine_question_type", route_by_question_type)
    builder.add_conditional_edges("determine_with_history", route_by_question_type)

    builder.add_edge("handle_entity_type", "check_entity_result")
    builder.add_edge("handle_knowledge_type", END)
    builder.add_edge("graph_retrieval", "recall_decision")
    builder.add_edge("text_retrieval", "recall_decision")
    builder.add_conditional_edges("recall_decision", route_after_recall)
    builder.add_edge("final_answer", END)

    return builder.compile()

# ========== agent 执行入口 ==========
async def run_agent(graph, llm, question: str, record: List[Dict] = None) -> str:
    initial_state: AgentState = {
        "question": question,
        "parsed_output": {},
        "step": 1,
        "database": None,
        "strategy": None,
        "context_history": {},
        "retrieved_chunks": [],
        "references": [],
        "sufficient": False,
        "llm": llm,
        "record": record,
        "ori_answer": None,
        "answer": None,
        "question_type": None,
        "retried_with_mixed": False,
        "original_question": question,
    }
    result = await graph.ainvoke(initial_state)
    return result
