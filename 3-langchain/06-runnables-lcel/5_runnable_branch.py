from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch, RunnableLambda
import os

load_dotenv()

def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template='Write a detailed explanation about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)


# model = ChatOpenAI()
model = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0.7,
    base_url="https://nqiyoamhphztywolejaq.supabase.co/functions/v1/gateway/openai/v1",
    api_key= os.environ.get('OPENAI_API_KEY')
)


parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Russia Vs Ukraine'}))