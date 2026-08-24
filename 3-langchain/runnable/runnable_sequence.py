from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# model = ChatOpenAI()

model = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0.7,
    base_url="https://nqiyoamhphztywolejaq.supabase.co/functions/v1/gateway/openai/v1",
    api_key= os.environ.get('GATEWAY_KEY')
)
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic': 'AI'}))