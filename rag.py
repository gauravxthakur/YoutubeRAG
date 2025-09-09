import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
#from langchain_pinecone import PineconeVectorStore

#from pytube import YouTube
#import openai
#import whisper
#import pinecone

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUTUBE_VIDEO = "https://youtu.be/dViKieKCM7o?si=fOMHR2oxIqJA4lGV"

# Setup the model
model = ChatOpenAI(model="deepseek/deepseek-chat-v3.1:free", openai_api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

# Define the parser
parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain  = prompt | model | parser

print(chain.invoke({
    "context": "Julius Caesar is my Maths teacher",
    "question": "Who is Julius Caesar?"
}))