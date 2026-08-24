from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem  - \n {poem}',
    input_variables=['poem']
)
parser = StrOutputParser()

cricket_txt = Path(__file__).parent / 'cricket.txt'

loader = TextLoader(
    cricket_txt,
    encoding='utf-8',
    autodetect_encoding=True,
)

docs = loader.load()

# print(docs[0])

chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})


print(result)

