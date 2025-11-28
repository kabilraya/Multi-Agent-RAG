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

collection_name_misumi = "Misumi Bearing Nuts"
collection_name_nsk = "NSK Bearing Nuts"

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

    return all_products

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

    return {"total_nsk":total_nsk, "total_misumi":total_misumi}

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

    Here are the list of your capabilities,
    1. I can use vector database collections to search for mechanical parts(known as part numbers) and their dimensions.
    2. I can count the number of mechanical parts or products I have from the database.
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
    1. First you MUST call the tool "retrieval" to retrieve the  releveant documents from the user query.
    2. Analyse the list of retrieved docs and generate the proper table format.
    3. Analyse the retrieved documents and see if the schema is different. The retrieved documents are a single row in "Part Number Name: value 1 | Price: value 2 | ... |".
    4. The retrieved documents can have different schema. So, if it is different you MUST divide the table and mention the **sub-category** field on top of each table generated.
    5. **Always use all the retrieved documents. Do not miss out on any of them.**

    ### Queries about the number of part numbers available (e.g. "How many part-numbers are available", "How many subcategories are there in Bearing lock nuts.")
    1. First you MUST call the tool "scrolling_function" to get the number of products availbale.
    2. It returns two numbers. You MUST ONLY generate the numbers retrieved. (e.g. "There are 107 subcategories in Misumi","There are 10 subcategories in NSK"). Here the numbers mentioned are just random. You MUST use the retrieved numbers from the tool.
    3. Just generate a raw text mentioning the number of sub categories.
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

    #Examples
    User: Who is the president of Nepal.
    Agent: I can answer questions only related to mechanical parts.
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
        if event.is_final_response():
            final_response = event.content.parts[0].text
            return final_response



