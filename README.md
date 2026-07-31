# GEN-AI Learning Repository

This repository is a hands-on learning workspace for building Generative AI applications with LangChain and related models. It contains beginner-friendly examples for working with LLMs, chat models, embeddings, and vector similarity.

## What you will learn

- Basics of LangChain and LangChain Core
- Integration with popular LLM providers such as OpenAI, Groq, Gemini, and Hugging Face
- Prompting and chat model usage
- Embedding generation and document similarity search
- Environment setup for AI application development

## Repository structure

- 3-langchain/Models/1.LLMs/ - simple LLM usage examples
- 3-langchain/Models/2.ChatModels/ - chat-based model examples for Groq, Gemini, OpenAI, and Hugging Face
- 3-langchain/Models/3.EmbeddingModels/ - embedding generation and similarity examples

## Prerequisites

- Python 3.9 or newer
- pip
- Internet access for API-based examples

## Setup

1. Clone the repository
2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Set up your environment variables

Create a `.env` file or export your API keys in the terminal, for example:

```bash
export OPENAI_API_KEY="your-openai-key"
export GROQ_API_KEY="your-groq-key"
export GOOGLE_API_KEY="your-google-key"
```

## Verify installation

Run the following command to confirm LangChain is installed correctly:

```bash
python verify_langchain.py
```

## Running examples

You can run the example scripts from the repository root, for example:

```bash
python "3-langchain/Models/2.ChatModels/3_chatmodel_openAI.py"
```

## Notes

- Some scripts require API keys from external providers.
- Make sure your environment variables are available before running those files.
- This repository is intended for learning and experimentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


## if got Import "langchain_huggingface" could not be resolved
Press:
Ctrl + Shift + P

Search for:
Python: Select Interpreter


## using chatmodel openai
https://www.youtube.com/watch?v=y5EmRr1O1h4&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0
https://github.com/campusx-official/langchain-structured-output/blob/main/with_structured_output_json.py
