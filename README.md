---
title: Groq RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Simple RAG Chatbot with Groq API

This is a Retrieval-Augmented Generation (RAG) chatbot built using Python, Streamlit, LangChain, and the Groq API. 

## Features
- Upload PDF documents.
- Automatically chunks and embeds the document using a local HuggingFace embedding model (`all-MiniLM-L6-v2`).
- Answers questions about your document utilizing lightning-fast inference from Groq's Llama 3 API.

## How to use
1. Get a free Groq API key from [console.groq.com](https://console.groq.com/).
2. Enter your API key in the sidebar.
3. Upload a PDF document and click "Process Document".
4. Chat with your document!
