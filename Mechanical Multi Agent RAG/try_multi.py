from google.adk.agents import LlmAgent, Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google import genai 
from google.genai import types
from google.adk.apps import App
from dotenv import load_dotenv
import os
from Misumi_Data.retrieval_misumi import retrieve_part_numbers
from NSK_Data.retrieval_nsk import part_numbers_retrieval
from qdrant_client import QdrantClient,models
import uuid
import asyncio
import ollama
import json

collection_name_misumi = "Misumi Bearing Nuts"
collection_name_nsk = "NSK Bearing Nuts"
ollama_client = ollama.Client(host='http://127.0.0.1:11434')
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")



# if not api_key:
#     raise ValueError("API key not found")
# genai.configure(api_key=api_key)

client = QdrantClient(url = "http://localhost:6333")

APP_NAME = "mechanical_parts_agent"






#Tools definition

def retrieval(base_query):
    """
    Retrieves the documents from two different collections according to the user query

    Args:
    base_query : The high level query provided by the user

    Returns:
        A list of retrieved docs from both vector database collections

    """
    all_products = []

    misumi_products = retrieve_part_numbers(query=base_query)
    nsk_products = part_numbers_retrieval(query=base_query)

    all_products.append(misumi_products)
    all_products.append(nsk_products)

    return json.dumps(all_products)

def scrolling_function():
    """
    Retrieves the number of products each website (Misumi and NSK) has from two different collections

    Args:
    None

    Returns:
    Returns two total numbers -> total_misumi and total_nsk
    """


    res_misumi, _ = client.scroll(
        collection_name= collection_name_misumi,
        limit = 1,
        with_payload = True,
        with_vectors = False,
        order_by = {
            "key" : "chunk_id",
            "direction" : "desc"
        }
    )
    product_number_misumi = res_misumi[0].payload.get("subcategory_number")
    total_misumi = product_number_misumi + 1

    res_nsk, _ = client.scroll(
        collection_name = collection_name_nsk,
        limit = 1,
        with_payload = True,
        with_vectors = False,
        order_by={
            "key" : "chunk_id",
            "direction" : "desc"
        }
    )
    product_number_nsk = res_nsk[0].payload.get("chunk_id")
    total_nsk = product_number_nsk + 1

    facet_response_misumi = client.facet(
    collection_name="Misumi Bearing Nuts",
    key="sub-category",
    exact = True,
    limit=100
    )

    result_misumi = {hit.value: hit.count for hit in facet_response_misumi.hits}

    facet_response_nsk = client.facet(
    collection_name="NSK Bearing Nuts",
    key="sub-category",
    exact = True,
    limit=150
    )

    result_nsk = {hit.value: hit.count for hit in facet_response_nsk.hits}

    data =  {
        "total_misumi": total_misumi,
        "total_nsk": total_nsk,
        "misumi_subcategories": result_misumi,
        "nsk_subcategories": result_nsk
    }

    return json.dumps(data)

def format_final_response(raw_response: str, author: str, original_query: str) -> str:
    """
    Formats the final response based ONLY on which agent generated it.
    """
    # If the response is from the PartNumberAgent, it needs table formatting.
    if author == "part_number_agent":
        print("[Formatter] PartNumberAgent detected. Sending to Ollama for table formatting.")
        
        # A generic prompt that instructs the local LLM how to handle either data type.
        # This removes the need for data checking in our Python code.
        prompt = f"""
        You are a data formatting assistant. Your task is to convert the following JSON data into a clean, readable Markdown format. Do not add any conversational text, headers, or explanations. Only output the formatted data.

        Analyze the structure of the JSON:
        - If the JSON contains keys like 'total_misumi' and 'subcategories', format it into a summary of totals and then create a Markdown table for each brand's subcategories.
        - If the JSON is a list of product objects, format the list into one or more Markdown tables. Group the products by their 'sub-category'.

        Based on the original user query: "{original_query}"

        JSON Data to format:
        {raw_response}
        """
        try:
            # Call the local Ollama model ONLY for this agent
            response = ollama_client.chat(
                model='gemma3:4b-it-qat',
                messages=[{'role': 'user', 'content': prompt}],
            )
            return response['message']['content']
        except Exception as e:
            return f"Error formatting data with local model: {e}\nRaw data: {raw_response}"
    
    # For all other agents, return their response as raw text.
    else:
        print(f"[Formatter] {author} detected. Returning raw text.")
        return raw_response

SelfDescriptionAgent = LlmAgent(
    name = "self_description_agent",
    model = "gemini-2.5-flash-lite",
    description="Describe the capabilities of the system",
    instruction="""
    You are a part of the system which contain an agent called "Part Number" which answers question based on part numbers which are mechanical parts,
    so your job is to answer the capabilities of the system, what it can do and how it works.

    #Examples
    User: What can you do?
    Agent: I can answer about mechanical parts.
    **Greet the user and mention your capabilities briefly**
    Here are the list of your capabilities,
    1. I can use vector database collections to search for mechanical parts(known as part numbers) and their dimensions.
    2. I can count the number of mechanical parts or products I have from the database.

    **Generate a more generic answer based on the capabilities and the example provided**
    """,
    output_key= "self_description_answers"
)

PartNumberAgent = LlmAgent(
    name = "part_number_agent",
    model = "gemini-2.5-flash-lite",
    description="The agent is responsible for answering about the mechanical parts known as part numbers",
    instruction= """
    You are a part of a system which generates or answers of the users asking about the mechanical parts. You are resposible for answering the queries of the users.
    # Your tasks:

    ## Analyse the user's query. Understand the user's intent on what the user is intending to know about.

    ### Queries about the part numbers (e.g. "List out different `bearing lock nuts`", "Tell me about `C-AN00`")
    1. First you MUST call the tool "retrieval" to retrieve the  relevent documents from the user query.
    2. **PRIMARY LOGIC GATE:** After retrieving the documents.
    3. Analyze the user's original query (e.g., "BNBSS25"). Is this query an **exact, case-insensitive, full-string match** for the value in the `Part Number Name` or `product_name` field of any of the documents you just retrieved?
    #### **SCENARIO A: If the answer to the check is YES (An exact match was found)**
    -   You are now in **"Specific Item Mode"**. Your ONLY goal is to return the single item that was matched.

    ---
    #### **SCENARIO B: If the answer to the check is NO (The query is a general category)**

    -   You are now in **"General List Mode"**. Your goal is to return all the items retrieved by the tool.
    
    ### Queries about the number of part numbers available (e.g. "How many products are available", "How many subcategories are there in Bearing lock nuts.")
    1. First you MUST call the tool "scrolling_function" to get the number of products availbale.
    The tool returns:
    - total_misumi
    - total_nsk
    - misumi_subcategories (dict)
    - nsk_subcategories (dict)

    **You MUST:**
    #### If the user is not asking for a specific website (e.g. `Misumi` or `NSK`) then just display every returned values in proper format.
    - Return for both the websites.

    #### If the user is asking about the specific website i.e. either "Misumi" or "NSK".
    -Return for just that website.
    """,
    tools=[scrolling_function, retrieval],
    output_key= "part_number_answers"
)

DefaultAgent = LlmAgent(
    name = "default_agent",
    model= "gemini-2.5-flash-lite",
    description="Give a default answer when the queries are unrelated to mechanical parts or greetings.",
    instruction="""
    You are a part of a system which answers about mechanical parts known as the `Part Numbers`. You are responsible for generating the default answers when the queries are unrelated to mechanical parts or greetings.

    
    
    **Mention that you don't know about the user's question and what you're capable of instead.**
    ** Do not just use the given example. Create more dynamic answers that you don't know about the question being asked.**
    """,
    output_key="default_answers"

)

router_agent = LlmAgent(
    name= "PartsRouter",
    model="gemini-2.5-flash-lite",
    description="The main router for the Part Number AI system",
    instruction="""
    You are a Part Number AI Assistant Router. Your job is to analyze the user's query and delegate the task to the appropriate sub-agent
    - If the user's query is a greeting (e.g., "Hi", "Hello") or asks about your capabilities(e.g., "What can you do?","Tell me about yourself"), delegate to the "SelfDescriptionAgent".
    - If the user's query is about part numbers (e.g., "Bearing lock nuts", "Tell me about X part number or subcategory") delegate to PartNumberAgent
    - If the user's query is about something other than the two mentioned above, delegate to the DefaultAgent.
    - Do not try to answer the user's questions directly. Your only task is to route the request to the correct sub-agent.
    """,
    sub_agents=[SelfDescriptionAgent,PartNumberAgent,DefaultAgent]
)
# app.register(router_agent)
app = App(
    name = APP_NAME,
    root_agent=router_agent,
    )

session_service = InMemorySessionService()

runner = Runner(app = app,session_service= session_service)
async def chat_with_agent(query):
    SESSION_ID = str(uuid.uuid4())
    USER_ID = str(uuid.uuid4())
    session = await session_service.create_session(app_name = APP_NAME, user_id = USER_ID, session_id =SESSION_ID)
    content = types.Content(role= 'user', parts = [types.Part(text= query)])
    events = runner.run(user_id=USER_ID,session_id=SESSION_ID,new_message=content)
    for event in events:
        print(event)
        if event.is_final_response():
            final_response = event.content.parts[0].text
            response_author = event.author
            



