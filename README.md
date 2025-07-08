# HealthCareVA_NVIDIA_GTM

**Health Care Virtual Assistant powered by NVIDIA and LangChain/LangGraph**

---

## Overview

HealthCareVA_NVIDIA_GTM is an advanced healthcare virtual assistant leveraging NVIDIA's LLMs, OpenAI, and retrieval-augmented generation (RAG) techniques. It integrates LangChain, LangGraph, and multiple toolchains (Arxiv, Wikipedia, Tavily, custom LLMs) to provide accurate, context-aware, and up-to-date medical information and conversational support.

---

## Features

- **Conversational AI**: Natural language chat with context retention and memory.
- **RAG (Retrieval-Augmented Generation)**: Integrates external knowledge sources (Arxiv, Wikipedia, Tavily, PDFs).
- **Multi-LLM Support**: Uses NVIDIA, OpenAI, and Palmyra-Med models.
- **Tool-Calling**: Dynamically invokes search and retrieval tools as needed.
- **Graph-based Workflows**: Modular, extensible conversation and tool orchestration using LangGraph.
- **Secure Deployment**: Environment variable and secret management best practices.
- **Logging & Exception Handling**: Robust error and activity logging.

---

## Project Structure
NVIDIA_Project/
├── finalapp.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── logger.py
    └── exception.py

## Setup Instructions

1. **Clone the repository**
   ```sh
   git clone https://github.com/nikhilsuresh45/HealthCareVA_NVIDIA_GTM.git
   cd NVIDIA_Project 
### Create a virtual environment

```sh
python3 -m venv nvidiavnev


###  Activate the virtual environment (for macOS/zsh)

```sh
source nvidiavnev/bin/activate

### Install dependencies
```sh
pip install -r requirements.txt

## Run the application
python finalapp.py 
