# Local Company Chatbot

This is a local chatbot application that serves both customers and employees with company data. The chatbot runs completely offline using Hugging Face models and LangChain for orchestration.

## Features

- 100% offline operation
- Uses local Hugging Face models
- LangChain for conversation management
- Vector database for efficient data retrieval
- FastAPI backend for easy integration
- Separate handling for customer and employee queries

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `data` directory and place your company documents there (PDFs, text files, etc.)

3. Run the application:
```bash
python main.py
```

The chatbot will be available at `http://localhost:8000`

## Usage

- Access the chatbot through the web interface
- Select whether you're a customer or employee
- Start chatting with the bot
- The bot will use the local knowledge base to answer questions

## Security

- All data is stored locally
- No internet connection required
- No data is sent to external servers 
