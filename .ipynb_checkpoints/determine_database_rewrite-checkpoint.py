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


def determine_database(query, llm):
    template = """
    You are an intelligent query analysis assistant.
    Your task is to determine which database should be accessed based on the user's question, and output a structured call instruction.
    
    Available databases:
    - Literature Text Database: containing extracted textual contents from literature in the field of nonlinear optics. It supports similarity-based retrieval based on query texts.
    - Literature Graph Database: containing structured information about specific entities in the literature, such as papers, authors, publication years, keywords and citation relationships.
    
    Task Requirements:
    1. Carefully analyze the user's question intent.
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
    - If using the Literature Graph Database, provide the question (or simplify) that needs to be retrieved from the graph database.
    - If using both, decide which database to call first, than give the corresponding content of the first database.
    
    Examples:
    
    Example 1:
    User Question: "What are the main mechanisms of third-harmonic generation in nonlinear optical materials?"
    Output:
    【Question Type】: Knowledge-Type 
    【Database to Call】: Literature Text Database 
    【First Database to Call】: N/A
    【Specific Call Strategy】: the main mechanisms of third-harmonic generation in nonlinear optical materials.
    
    Example 2:
    User Question: "Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?"
    Output:
    【Question Type】: Entity-Type
    【Database to Call】: Literature Graph Database
    【First Database to Call】: N/A
    【Specific Call Strategy】: Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?
    
    Example 3:
    User Question: "What are the potential applications of Kerr optical frequency comb in the field of quantum computing and what literature supports these applications?"
    Output:
    【Question Type】: Mixed-Type 
    【Database to Call】: Both
    【First Database to Call】: Literature Graph Database
    【Methods】：
    First retrieve papers of keywords kerr optical frequency comb and quantum computing from graph database.
    If have, summarize the abstracts of the papers to answer the question. 
    If not, retrieve the textual content from text database to get relative texts and the source paper titles and answer the question.
    【Specific Call Strategy】: What are the potential applications of Kerr optical frequency comb in the field of quantum computing and what literature supports these applications?
    
    Example 4:
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
    
    User Question: {question}
    """
    
    prompt_template = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format_messages(question=query)
    response = llm.invoke(formatted_prompt)

    parsed_output, is_valid = parse_and_check_llm_output(response.content)
    return parsed_output, response


def extract_recall(text):
    # Initialize the dictionary to store the extracted information
    extracted_info = {
        "sufficient": None,
        "next database": None,
        "strategy": None,
        "reason": None,
    }

    # Use regex to extract each component from the response text
    sufficient_match = re.search(r"【Sufficient】\s*[:：]\s*(True|False)", text, re.IGNORECASE)
    next_db_match = re.search(r"【Next Database to Call】\s*[:：]\s*(.+)", text, re.IGNORECASE)
    reason_match =  re.search(r"【Reason】\s*[:：]\s*([\s\S]+?)(?=【|$)", text, re.IGNORECASE)
    strategy_match = re.search(r"【Specific Call Strategy】\s*[:：]\s*([\s\S]+?)(?=【|$)", text, re.IGNORECASE)

    # Populate the dictionary based on matches
    if sufficient_match:
        extracted_info["sufficient"] = sufficient_match.group(1) == "True"
    
    if next_db_match:
        extracted_info["next database"] = next_db_match.group(1).strip()
    
    if strategy_match:
        extracted_info["strategy"] = strategy_match.group(1).strip()

    if reason_match:
        extracted_info["reason"]  = reason_match.group(1).strip()

    # Return the result as a JSON-compatible dictionary
    return extracted_info


def database_recall(llm, question, context, strategy):
    template = '''
    You are an intelligent query analysis assistant. You will be provided with a question, some contexts and a specific database call strategy. 
    Your task is to determine if the contexts already retrieved is sufficient to answer the question. If not, determine which database should be accessed next based on the user's question, the contexts already retrieved from one of them and the specific call strategy.
    If the strategy reveal that the graph database have been retrieved and the contexts is empty, use another database.
    DO NOT call the same database twice in a row. If the previous database called is text database, use the contexts to make an entity dict to call graph database. If previous database is graph database, use the contexts to retrieve texts and source paper titles from text database.
   
    Available databases:
    - Literature Text Database: containing extracted textual contents from literature in the field of nonlinear optics. It supports similarity-based retrieval based on query texts.
    - Literature Graph Database: containing structured information about specific entities in the literature, such as papers, authors, publication years, keywords and citation relationships.

     Task Requirements:
    1. Carefully analyze the user's question intent, the given strategy and the contexts.
    2. Decide if the contexts is sufficient to answer the question. If is, let sufficient be True. 
    And if 【Sufficient】 is True, only output 【Sufficient】, because the other information are not needed.
    3. If sufficient is False, additionally determine:
       - Which database should be accessed next (Literature Graph Database or Literature Text Database);
       - Why the contexts is not sufficient.
    4. Output your response strictly following this format:
    
    【Sufficient】: True/False
    【Next Database to Call】: Literature Text Database/Literaure Graph Database
    【Reason】: (if not sufficient) The reason why the context is not sufficient for answering the question.
    【Specific Call Strategy】: 
    - If using the Literature Text Database, only provide the textual content that needs to be used for similarity retrieval (NOT question, but the key contents extracted from the question). Do not include any additional explanations.
    - If using the Literature Graph Database, just provide a dict of entities that need to be retrieved from graph database.(some elements can be empty)
    Dict format: {{'Authors': [], 'Papers': [], 'Years': [], 'Sources': []}}

    Question: {question}
    
    Contexts: {context}
    
    Strategy: {strategy}
    '''
    prompt_template = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format_messages(question=question, context=context, strategy=strategy)
    response = llm.invoke(formatted_prompt)
    parsed_output = extract_recall(response.content)

    return parsed_output, response


def final_answer(llm, question, contexts, strategy):
    template = '''
    You are an intelligent query assistant in the area fo optical frequency comb. 
    You will be provided with a question, some contexts retrieved from two databases and the specific database call strategy.

    The two available databases:
    - Literature Text Database: containing extracted textual contents from literature in the field of nonlinear optics. It supports similarity-based retrieval based on query texts.
    - Literature Graph Database: containing structured information about specific entities in the literature, such as papers, authors, publication years, keywords, citation relationships etc.

    The contexts are retrieved fully follow the specific database call strategy. Your task is to decide which contexts are useful in answer the question, than generate the answer of the question mainly based on the contexts and strategy provided. Also you can incorporate analysis appropriately.
    DO NOT mention phrases such as "in the context", "the chunk", "the database" or similar wording. 
    Provide detailed, factual, and well-organized answers.

    Question: {question}
    
    Contexts: {contexts}
    
    Strategy: {strategy}
    '''
    
    prompt_template = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format_messages(question=question, contexts=contexts, strategy=strategy)
    response = llm.invoke(formatted_prompt)
    
    return response.content