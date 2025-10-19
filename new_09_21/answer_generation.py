from langchain.prompts import ChatPromptTemplate
from retriever_weaviate import retriever_weaviate

def answer_generation(llm, context, question):
    template = '''
    Your task is to generate detailed answers to the given questions based on specific paragraphs from scientific papers in the field of optical frequency combs.
    You should think step by step, first decide which texts are helpful in answering the question, then use these useful texts to answer the given questions.
    
    Context: You will be provided with scientific paragraphs from papers in optical frequency combs.  
    These paragraphs contain key insights, findings, or theoretical details relevant to the topic.
    
    Objective: Your goal is to generate comprehensive answers to each question completely based on the context provided.  
    Each answer should address the specific question thoroughly.
    
    Style: Write the responses in an academic style, using precise technical language and adhering to the conventions of scientific writing.  
    Ensure that your answers are factual, structured, and clear.
    
    Tone: The tone should be professional, suitable for researchers in the field of nonlinear optics.
    
    Audience: Your answers should be tailored for researchers in nonlinear optics, 
    so please assume a high level of prior knowledge and familiarity with advanced concepts in the field.
    
    Response Format: Provide segmented, clear, and well-organized answers.  Each response should directly address the question and be logically structured.  
    Use appropriate headings and subheadings if needed for clarity. Do not mention phrases such as "in the context", "in the picture", or similar wording. 
    Contexts: {context}
    Question: {question}
    Answer:'''

    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(question=question, context=context)
    response = llm.invoke(formatted_prompt)
    return response.content


def chunk_output(chunks):
    for idx, chunk in enumerate(chunks):  
        print(f"Chunk {idx + 1}: {chunk['chunk']}")
        title = chunk.get('title', 'No title available')  
        print(f"source paper: {title}")
        
        refs = chunk.get('references', {})  
        figs = chunk.get('figs', [])       
        
        if refs:
            print("references:")
            for k, v in refs.items():
                print(f"[{k}] {v}")
        else:
            print("No reference mentioned")
        
        if figs:
            print("figs:")
            for fig in figs:
                figname = fig.get('img_path', 'No image path available')
                print(f"Fig {fig.get('fig_num', 'N/A')}: {fig.get('fig_content', 'No content available')}")
                show_image(figname)
        else:
            print("No figs mentioned")
        
        print("")