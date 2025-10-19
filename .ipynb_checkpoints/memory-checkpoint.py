from langchain.prompts import ChatPromptTemplate
import re
import json


def parse_and_check_llm_output(text: str):
    """
    Parse the LLM output and check completeness.
    Returns structured fields and a boolean indicating whether it passed the check.
    """
    # Parse fields
    parsed = {
        "question_type": None,
        "database_to_call": None,
        "first_database_to_call": "N/A",
        "methods": None,
        "call_strategy": None
    }
    
    matches = {
        "question_type": re.search(r"【Question Type】\s*[:：]\s*(.+)", text, re.IGNORECASE),
        "database_to_call": re.search(r"【Database to Call】\s*[:：]\s*(.+)", text, re.IGNORECASE),
        "first_database_to_call": re.search(r"【First Database to Call】\s*[:：]\s*(.+)", text, re.IGNORECASE),
        "methods": re.search(r"【Methods】\s*[:：]\s*([\s\S]+?)(?=【|$)", text, re.IGNORECASE),
        "call_strategy": re.search(r"【Specific Call Strategy】\s*[:：]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    }

    for key, match in matches.items():
        if match:
            parsed[key] = match.group(1).strip()
    
    # Check completeness
    pass_check = (
        parsed["question_type"] in ["Knowledge-Type", "Entity-Type", "Mixed-Type"] and
        parsed["database_to_call"] in ["Literature Text Database", "Literature Graph Database", "Both"] and
        parsed["call_strategy"] is not None
    )

    # Additional check for Mixed-Type
    if parsed["question_type"] == "Mixed-Type":
        pass_check = pass_check and parsed["first_database_to_call"] in ["Literature Text Database", "Literature Graph Database"] and parsed["methods"] != "N/A"
    
    return parsed, pass_check
    

def determine_database_with_history(query, llm, record):
    template = """
    You are an intelligent query analysis assistant.
    Your task is to determine which database should be accessed based on the history record and new question, and output a structured call instruction. The history record are the the previous questions asked by the user and the answers. 
    
    Available databases:
    - Literature Text Database: containing extracted textual contents from literature in the field of nonlinear optics. It supports similarity-based retrieval based on query texts.
    - Literature Graph Database: containing structured information about specific entities in the literature, such as papers, authors, publication years, keywords and citation relationships.
    
    Task Requirements:
    1. Carefully analyze the user's question intent and useful information of the history record.
    2. Classify the question into one of the following categories:
       - Knowledge-Type: related to domain knowledge, theoretical summaries, experimental methods, etc. → use the Literature Text Database.
       - Entity-Type: involving specific entities like specific papers(include abstract), authors, years, keywords, citation relationships. → use the Literature Graph Database. 
       - Mixed-Type: requiring both entity identification and content retrieval → use both databases sequentially.
    3. If the question is Mixed-Type, additionally determine:
       - Which database should be accessed first (Literature Graph Database or Literature Text Database);
    4. Output your response strictly following this format:
    
    【Question Type】: (Knowledge-Type / Entity-Type / Mixed-Type)
    【Database to Call】: (Literature Text Database / Literature Graph Database / Both)
    【First Database to Call】: (only for Mixed-Type, otherwise N/A)
    【Methods】: (only for Mixed-Type) Give a specific call strategy step by step. You must take into account that the papers of the specified keywords may not be found in the graph database.
    【Specific Call Strategy】:
    - If using the Literature Text Database, directly provide the textual content that needs to be used for similarity retrieval. Do not include any additional explanations.
    - If using the Literature Graph Database, provide the question that needs to be retrieved from the graph database.
    - If using both, decide which database to call first, than give the corresponding content of the first database. 

    Since there may be some useful information in the history record, please read the history record first than generate the Methods and Specific Call Strategy.The history record may contain some useful information. If the new question contains words like "the paper" "the concept" or similar words, find them in the history record and replace them with specific information.
    
    
    Examples:
    
    Example 1:
    History Record:
    {{'user': What is third-harmonic generation in nonlinear optical materials?  'assistant': ...}}
    User Question: "What are the main mechanisms of it?"
    Output:
    【Question Type】: Knowledge-Type 
    【Database to Call】: Literature Text Database 
    【First Database to Call】: N/A
    【Specific Call Strategy】: the main mechanisms of third-harmonic generation in nonlinear optical materials.
    
    Example 2:
    History Record:{{}}
    User Question: "Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?"
    Output:
    【Question Type】: Entity-Type
    【Database to Call】: Literature Graph Database
    【First Database to Call】: N/A
    【Specific Call Strategy】: Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?

    Example 3:
    History Record:
    {{'user': Who wrotes Dissipative Kerr solitons in optical microresonators, 'assistant': ...}}
    User Question: "Which papers cites this paper?"
    Output:
    【Question Type】: Entity-Type
    【Database to Call】: Literature Graph Database
    【First Database to Call】: N/A
    【Specific Call Strategy】: Which papers cites Dissipative Kerr solitons in optical microresonators?
    
    Example 4:
    History Record: {{}}
    User Question: "Which papers on the temperature stability of Kerr frequency combs are widely cited?"
    Output:
    【Question Type】: Mixed-Type 
    【Database to Call】: Both
    【First Database to Call】: Literature Graph Database
    【Methods】:
    First retrieve papers of keywords kerr frequency comb and temperature stability from graph database.
    If have, rank the papers based on citations and answer the question. 
    If not, retrieve the textual content from text database to get relative texts and paper titles.
    Then search the titles in the graph database for citations and ranked them.
    【Specific Call Strategy】: Which papers on the temperature stability of Kerr frequency combs are widely cited?
    
    History Record: {record}
    
    User Question: {question}
    """
    
    prompt_template = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format_messages(question=query, record=record)
    response = llm.invoke(formatted_prompt)

    parsed_output, is_valid = parse_and_check_llm_output(response.content)
    return parsed_output, response

