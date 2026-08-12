# AI Document Q&A Knowledge Assistant

Upload any PDF and chat with it. Ask questions in plain English and get answers grounded in the document's actual content — with source snippets shown for every answer, so you can verify where the response came from.

Built entirely on free, open tools — no paid API keys required to run it.

## What it does

- Upload a PDF through the browser (no manual file placement needed)
- Automatically splits the document into overlapping chunks for better context retrieval
- Generates semantic embeddings locally on your machine — no external API call, no cost
- Retrieves the most relevant chunks using MMR (Maximal Marginal Relevance) for diverse, non-redundant context
- Sends that context to a free, fast LLM (Groq) to generate a grounded answer
- Displays the exact source chunks behind each answer for transparency
- Persists chat history in a proper chat-style interface across a session

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free) |
| Vector store | FAISS |
| LLM | Groq (`llama-3.1-8b-instant`, free tier) |
| Orchestration | LangChain |

## Why this setup

Most RAG tutorials default to OpenAI for both embeddings and generation, which means you can't run or demo the project without a paid API key. This version swaps that out for a fully free stack: embeddings run locally on CPU, and generation uses Groq's free tier — so the whole thing is reproducible by anyone without spending money.

## Getting started

```bash
git clone https://github.com/mahi04git/ai-document-qa-assistant.git
cd ai-document-qa-assistant

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_free_groq_key
```

Get a free key at [console.groq.com](https://console.groq.com).

Run the app:

```bash
streamlit run main.py
```

## Using it

1. Open the app in your browser (Streamlit will print the local URL)
2. Upload a PDF from the sidebar and click **Process Document**
3. Ask questions in the chat box at the bottom
4. Expand **Sources** under any answer to see the exact text it was grounded in

## Possible next steps

- Support multiple documents in one session
- Add a reranking step for better retrieval precision
- Deploy to Streamlit Community Cloud for a live demo link
- Add evaluation metrics (retrieval recall, answer faithfulness)

## Credits

Started from an open-source RAG starter project and rebuilt the embeddings/LLM layer, upload flow, and chat UI on top of it.

