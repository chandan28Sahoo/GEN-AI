# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import os

# load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V4-Flash",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
# )

# chat_model = ChatHuggingFace(llm=llm)


chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Please explain in simple terms, what is {topic}.'),
    # SystemMessage(content='You are a helpful {domain} expert.'),
    # HumanMessage(content='Please explain in simple terms, what is {topic}.'),
])


prompt = chat_template.invoke({
    'domain': 'AI',
    'topic': 'LangChain'
})

print(f"Prompt: {prompt}")