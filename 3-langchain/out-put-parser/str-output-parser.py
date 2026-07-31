from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)

# 1st prompt -> details report
template1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write 5 lines summary on following text: /n {text}",
    input_variables=["text"]
)

# prompt1 = template1.invoke({"topic": "Samsung Galaxy S24 Ultra"})
# response1 = chat_model.invoke(prompt1)

# prompt2 = template2.invoke({"text": response1.content})
# response2 = chat_model.invoke(prompt2)

# print("Detailed Report:\n", response1.content)
# print("\nSummary:\n", response2.content)

parser = StrOutputParser()
chain = template1 | chat_model | parser | template2 | chat_model | parser


response = chain.invoke({"topic": "Samsung Galaxy S24 Ultra"})

print("Detailed Report:\n", response)