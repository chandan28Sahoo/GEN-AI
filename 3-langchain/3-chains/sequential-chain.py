from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)

prompt_template1 = PromptTemplate(
    template="Generate a detailed report On {topic}",
    input_variables=["topic"]
)

prompt_template2 = PromptTemplate(
    template="Generate a 5-point summary from the following text \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt_template1 | chat_model | prompt_template2 | chat_model | parser
response = chain.invoke({"topic": "Unemployement in India"})

print(response)
chain.get_graph().print_ascii()
