# 🎥 YouTube RAG Chatbot with LangChain and Pinecone

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system built in Python that can answer questions based on the transcript of a specific YouTube video. It automates the entire process: downloading the video's audio, transcribing it, indexing the text, and querying the data using a Large Language Model (LLM).

## ✨ Features

* **Audio Download & Caching:** Uses `yt-dlp` to download only the audio stream of a YouTube video and implements a caching mechanism to avoid re-downloading/re-transcribing.
* **Offline Transcription:** Leverages the open-source **Whisper** model for accurate speech-to-text conversion.
* **Vector Database:** Utilizes **Pinecone** to store and efficiently retrieve vector embeddings of the transcript chunks.
* **Embeddings:** Employs the fast and powerful `all-MiniLM-L6-v2` model via **HuggingFaceEmbeddings** for local vector generation.
* **RAG Pipeline:** Built with **LangChain**, creating a robust chain that retrieves relevant context from Pinecone and passes it to an LLM (via OpenRouter) for final answer generation.

## ⚙️ Prerequisites

Before running the script, ensure you have the following installed and configured:

1.  **Python 3.8+**
2.  **API Keys** for:
    * **OpenRouter** (for the LLM access)
    * **Pinecone** (for the vector database)
3.  **FFmpeg:** Required by `yt-dlp` for audio extraction and conversion.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gauravxthakur/YoutubeRAG.git
    cd YoutubeRAG
    ```

2.  **Install dependencies:**

    ```bash
    pip install python-dotenv langchain langchain-openai langchain-core langchain-huggingface langchain-pinecone yt-dlp openai-whisper
    ```


## 🔑 Setup

You must create a file named `.env` in the root directory of the project to store your sensitive keys.

**.env file structure:**

```env
OPENROUTER_API_KEY="your-openrouter-key"
PINECONE_API_KEY="your-pinecone-key"
# Optional: PINECONE_ENVIRONMENT="your-pinecone-env-if-needed"
