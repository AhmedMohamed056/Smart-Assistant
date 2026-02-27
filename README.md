# Smart Assistant - Context-Aware RAG System

A domain-specific Retrieval-Augmented Generation (RAG) engine designed to automate customer support for E-commerce businesses. This system allows users to upload policy documents (PDFs) to create an instant AI assistant that answers questions accurately based **strictly** on the provided documents.

## 🌟 Key Features

* **Context-Awareness (Conversational Memory):** The assistant understands follow-up questions and references to previous context (e.g., remembering which product or tier was discussed).
* **Zero-Hallucination Policy:** The system is configured with `temperature=0.0` and strict prompts to ensure it only answers using the provided context, preventing the invention of false information.
* **Fast Semantic Search:** Uses advanced HuggingFace embeddings (`bge-small-en-v1.5`) for rapid and accurate retrieval.
* **User-Friendly Interface:** Built with Streamlit for a smooth, interactive chat experience.

## 🛠️ Tech Stack

* **LLM:** Google Gemini 2.5 Flash
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Vector Database:** ChromaDB
* **Orchestration:** LangChain
* **Frontend:** Streamlit

## 🚀 Getting Started

### Prerequisites

* **Python 3.10 or 3.11** (Strictly required to avoid dependency conflicts with ChromaDB/Pydantic).
* A Google Gemini API Key.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/AhmedMohamed056/Smart-Assistant.git](https://github.com/AhmedMohamed056/Smart-Assistant.git)
    cd Smart-Assistant
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -U -r requirements.txt
    ```

4.  **Set Up API Key:**
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY="your_api_key_here"
    ```

### Running the Application

1.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```
2.  **Upload & Chat:**
    * Open the URL provided in your terminal (usually `http://localhost:8501`).
    * Upload your policy PDFs via the sidebar.
    * Click "Process Documents".
    * Start asking questions based on your documents!

## 🎥 Video Demo

[Watch the Demo on YouTube](https://youtu.be/s1NmMdHsQ9c)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.