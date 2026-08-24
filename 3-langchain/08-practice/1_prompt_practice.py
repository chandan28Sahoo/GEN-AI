from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


template1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

prompt1 =  template1.invoke({"topic": "Samsung Galaxy S24 Ultra"})

print("Prompt 1:\n", prompt1)