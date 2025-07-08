import streamlit as st
import os
import asyncio
from src.exception import CustomException
import sys
from src.logger import logger    
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun 
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage , AnyMessage , ToolMessage 
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS 
from pydantic import BaseModel, Field 
from typing import Optional  , Annotated 
from typing_extensions import TypedDict 
from langgraph.graph import StateGraph, START, END  
from langgraph.prebuilt import ToolNode , tools_condition 
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.graph.message import add_messages  
from IPython.display import display, Image 
from dotenv import load_dotenv 
load_dotenv() 

## Load the Nvidia API Key

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")  # type: ignore
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # type: ignore 
os.environ["NVIDIA_PALMYRA_API_KEY"] = os.getenv("NVIDIA_PALMYRA_API_KEY") # type: ignore 
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") # type: ignore 

llm=ChatNVIDIA (model="meta/llama-4-maverick-17b-128e-instruct") 
llm_med = ChatNVIDIA(model="writer/palmyra-med-70b")  # type: ignore 
llm_open_ai = ChatOpenAI(model="gpt-4o")


#  Agentic Tools 
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 3, doc_content_chars_max = 800) # type:ignore 
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, llm=llm_open_ai)  # type: ignore 
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 3, doc_content_chars_max = 800) # type:ignore 
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, llm=llm_open_ai)  # type: ignore  
tavily = TavilySearch()


# Using Palmyra LLM as a tool 
def llm_med_tool(input_text: str) -> str:
    return llm_med.invoke(input_text).content # type: ignore
 
llm_med_tool_obj = Tool(
    name="Palmyra-Med",
    func=llm_med_tool,
    description="Use this tool for medical or general questions. Input should be a string."
)


# Defining the Graph State  
class State(TypedDict): # type: ignore
    messages: Annotated[list[AnyMessage], add_messages] 

try:
    # Tools or Agents 

    tools = [llm_med_tool_obj, arxiv, tavily, wiki] 

    llm_with_tools = llm_open_ai.bind_tools(tools) 


    ### Defining the Nodes  
    def tool_calling_llm(state:State):
        return {"messages" : [llm_with_tools.invoke(state["messages"])]} # type:ignore   


    ## Building Graph 
    graph = StateGraph(State) 
    graph.add_node("tool_calling_llm", tool_calling_llm) 
    graph.add_node("tools", ToolNode(tools)) 


    # Defining Edges
    graph.add_edge(START, "tool_calling_llm") 
    graph.add_conditional_edges(
        "tool_calling_llm" , tools_condition
    )
    graph.add_edge ("tools", "tool_calling_llm")  


    # Compiling the Graph with MEMORY 
    memory= MemorySaver()
    graph_memory = graph.compile(checkpointer= memory)  # this is change i.e. WITH MEMORY 

    ## Specify the thread 

    config = {"configurable" : {"thread_id" : "2"}}  

    # Defining 

    # async def run_events():
    #     async for event in graph_memory.astream_events({'messages' : "Hello, i would like to know the contents of SARs Vaccine developed in Russia ? "},
    #                                                     config=config , version = "v2"):
    #         print ("EVENT", event) 


    # asyncio.run(run_events()) 

    messages = graph_memory.invoke(
        {"messages" : HumanMessage(content= " What are the latest development about Covid and its vaccine ? and what are the contents of this vaccine?")} , 
        config = config) 
    

    for msg in messages["messages"]:
         msg.pretty_print()   

except Exception as e:
    logger.error(f"An error occurred while creating security group: {e}")
    raise CustomException(e, sys) from e # type: ignore


# /************************************************************************************************************************/



    