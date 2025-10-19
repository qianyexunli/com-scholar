determine_prompt = '''
    You are an intelligent query analysis assistant.
    Your task is to determine which database should be accessed based on the user's question, and output a structured call instruction.
    
    Available databases:
    - Literature Text Database: containing extracted textual contents from literature in the field of nonlinear optics. It supports similarity-based retrieval based on query texts.
    - Literature Graph Database: containing structured information about specific entities in the literature, such as papers, authors, publication years, keywords and citation relationships.
    
    Task Requirements:
    1. Carefully analyze the user's question intent.
    2. Classify the question into one of the following categories:
       - Knowledge-Type: related to domain knowledge, theoretical summaries, experimental methods, etc. → use the Literature Text Database.
       - Entity-Type: involving specific entities like papers(include abstract), authors, years, keywords, citation relationships. → use the Literature Graph Database. 
       - Mixed-Type: requiring both entity(mainly papers) identification and content retrieval → use both databases sequentially.
       If the question only requires to search the Literature Text Database and do not mention entities like papers, authors, years, Just classify it into Knowledge-Type, instead of Mixed-Type.
    3. If the question is Mixed-Type, additionally determine:
       - Which database should be accessed first (Literature Graph Database or Literature Text Database);
    4. Output your response strictly following this json format:
    {{
    "Question Type": (Knowledge-Type / Entity-Type / Mixed-Type),
    "Database to Call":(Literature Text Database / Literature Graph Database / Both) 
    "First Database to Call":(only for Mixed-Type, otherwise N/A) ,
    "Methods": (only for Mixed-Type) Give a specific call strategy step by step. You must take into account that the papers of the specified keywords may not be found in the graph database.,
    "Specific Call Strategy":  - If using the Literature Text Database, directly provide the textual content that needs to be used for similarity retrieval. Do not include any additional explanations.
    - If using the Literature Graph Database, provide the question (or simplify) that needs to be retrieved from the graph database.
    - If using both, decide which database to call first, than give the corresponding content of the first database.
    }}
    
    Examples:
    
    Example 1:
    User Question: "What are the main mechanisms of third-harmonic generation in nonlinear optical materials?"
    Output:
    {{
    "Question Type": "Knowledge-Type",
    "Database to Call": "Literature Text Database",
    "First Database to Call": "N/A",
    "Specific Call Strategy": "the main mechanisms of third-harmonic generation in nonlinear optical materials."
    }}
    
    Example 2:
    User Question: "Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?"
    Output:
    {{
    "Question Type": "Entity-Type",
    "Database to Call": "Literature Graph Database",
    "First Database to Call": "N/A",
    "Specific Call Strategy": "Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?"
    }}
    
    Example 3:
    User Question: "What are the potential applications of Kerr optical frequency comb in the field of quantum computing and what literature supports these applications?"
    Output:
    {{
    "Question Type": "Mixed-Type",
    "Database to Call": "Both",
    "First Database to Call": "Literature Graph Database",
    "Methods": "First retrieve papers of keywords kerr optical frequency comb and quantum computing from graph database. If have, summarize the abstracts of the papers to answer the question. If not, retrieve the textual content from text database to get relative texts and the source paper titles and answer the question.",
    "Specific Call Strategy": "What are the potential applications of Kerr optical frequency comb in the field of quantum computing and what literature supports these applications?"
    }}
    
    Example 4:
    User Question: "Which papers on the temperature stability of Kerr frequency combs are widely cited?"
    Output:
    {{
    "Question Type": "Mixed-Type",
    "Database to Call": "Both",
    "First Database to Call": "Literature Graph Database",
    "Methods": "First retrieve papers of keywords kerr frequency comb and temperature stability from graph database. If have, rank the papers based on citations and answer the question. If not, retrieve the textual content from text database to get relative texts and paper titles. Then search the titles in the graph database for citations and ranked them.",
    "Specific Call Strategy": "Which papers on the temperature stability of Kerr frequency combs are widely cited?"
    }}
    
    User Question: {question}
'''

determine_history_prompt = '''
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
    4. Output your response strictly following this json format:
    
    {{
    "Question Type": (Knowledge-Type / Entity-Type / Mixed-Type),
    "Database to Call":(Literature Text Database / Literature Graph Database / Both) 
    "First Database to Call":(only for Mixed-Type, otherwise N/A) ,
    "Methods": (only for Mixed-Type) Give a specific call strategy step by step. You must take into account that the papers of the specified keywords may not be found in the graph database.,
    "Specific Call Strategy":  - If using the Literature Text Database, directly provide the textual content that needs to be used for similarity retrieval. Do not include any additional explanations.
    - If using the Literature Graph Database, provide the question (or simplify) that needs to be retrieved from the graph database.
    - If using both, decide which database to call first, than give the corresponding content of the first database.
    }}

    Since there may be some useful information in the history record, please read the history record first than generate the Methods and Specific Call Strategy.The history record may contain some useful information. If the new question contains words like "the paper" "the concept" or similar words, find them in the history record and replace them with specific information.
    
    
    Examples:
    
    Example 1:
    History Record:
    {{'user': What is third-harmonic generation in nonlinear optical materials?  'assistant': ...}}
    User Question: "What are the main mechanisms of it?"
    Output:
    {{
    "Question Type": "Knowledge-Type",
    "Database to Call": "Literature Text Database",
    "First Database to Call": "N/A",
    "Specific Call Strategy": "the main mechanisms of third-harmonic generation in nonlinear optical materials."
    }}
    
    Example 2:
    History Record:{{}}
    User Question: "Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?"
    Output:
    {{
    "Question Type": "Entity-Type",
    "Database to Call": "Literature Graph Database",
    "First Database to Call": "N/A",
    "Specific Call Strategy": "Which researchers have published papers on soliton dynamics in optical frequency combs since 2020?"
    }}

    Example 3:
    History Record:
    {{'user': Who wrotes Dissipative Kerr solitons in optical microresonators, 'assistant': ...}}
    User Question: "Which papers cites this paper?"
    Output:
    {{
    "Question Type": "Entity-Type",
    "Database to Call": "Literature Graph Database",
    "First Database to Call": "N/A",
    "Specific Call Strategy": "Which papers cites Dissipative Kerr solitons in optical microresonators?"
    }}
    
    Example 4:
    History Record: {{}}
    User Question: "Which papers on the temperature stability of Kerr frequency combs are widely cited?"
    Output:
    {{
    "Question Type": "Mixed-Type",
    "Database to Call": "Both",
    "First Database to Call": "Literature Graph Database",
    "Methods": "First retrieve papers of keywords kerr frequency comb and temperature stability from graph database. If have, rank the papers based on citations and answer the question. If not, retrieve the textual content from text database to get relative texts and paper titles. Then search the titles in the graph database for citations and ranked them.",
    "Specific Call Strategy": "Which papers on the temperature stability of Kerr frequency combs are widely cited?"
    }}
    
    History Record: {record}
    
    User Question: {question}
'''    
recall_prompt = '''
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
    And if "Sufficient" is True, only output "Sufficient", because the other information are not needed.
    3. If sufficient is False, additionally determine:
       - Which database should be accessed next (Literature Graph Database or Literature Text Database);
       - Why the contexts is not sufficient.
    4. Output your response strictly following this json format:

    {{
    "Sufficient": (True/False),
    "Next Database to Call":(Literature Text Database / Literature Graph Database) 
    "Reason": (if not sufficient) The reason why the context is not sufficient for answering the question.
    "Specific Call Strategy":  - If using the Literature Text Database, directly provide the textual content that needs to be used for similarity retrieval. Do not include any additional explanations.
     - If using the Literature Text Database, only provide the textual content that needs to be used for similarity retrieval (NOT question, but the key contents extracted from the question). Do not include any additional explanations.
    - If using the Literature Graph Database, just provide a dict of entities that need to be retrieved from graph database.(some elements can be empty) 
    Dict format: {{'Authors': [], 'Papers': [], 'Years': [], 'Sources': []}}
    }}
    Dict format: {{'Authors': [], 'Papers': [], 'Years': [], 'Sources': []}}

    Question: {question}
    
    Contexts: {context}
    
    Strategy: {strategy}
'''

