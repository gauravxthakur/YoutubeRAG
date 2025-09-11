import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

import yt_dlp
import whisper
import pinecone

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
YOUTUBE_VIDEO = "https://youtu.be/dViKieKCM7o?si=fOMHR2oxIqJA4lGV"

# Setup the model
model = ChatOpenAI(model="deepseek/deepseek-chat-v3.1:free", api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# Define the parser
parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Getting the transcript of the video
def get_video_transcription(video_url, cache_file="transcription.txt"):
    # check if a transcript file already exists
    if os.path.exists(cache_file):
        with open(cache_file,"r") as file:
            print("Using cached transcription...")
            return file.read()
        
    try:
        # Define the directory on the D drive and create it if it doesn't exist
        download_dir = "D:\\YT_DL_Downloads"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        # Define the yt-dlp options to download only audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(download_dir, '%(id)s.%(ext)s'),
            'keepvideo': False,
            'restrictfilenames': True,
        }

        # Use yt_dlp to download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading audio to D drive...")
            info = ydl.extract_info(video_url, download=True)
            
            # Update the path to point to the .mp3 file
            audio_path = os.path.join(download_dir, f"{info['id']}.mp3")


        # Load the whisper model and transcribe the audio
        model = whisper.load_model("base")
        transcription = model.transcribe(audio_path, fp16=False)

        # Save the transcription to a cache file
        with open(cache_file, "w") as file:
            file.write(transcription["text"])
        
        # Manually delete the temporary audio file
        os.remove(audio_path)
        
        return transcription["text"]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Get the transcription
transcription = get_video_transcription(YOUTUBE_VIDEO)


# Split the transcription into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents([Document(page_content=transcription)])


# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Set up Pinecone (vector db)
index_name = "youtube-rag-index"

pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)


# The full RAG chain
chain = ({"context": pinecone.as_retriever(), "question": RunnablePassthrough()} | prompt | model | parser)


print(chain.invoke("what are some of the words spoken in this video?...Also repeat my query at the end of your response, but in all caps!"))