from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate,load_prompt
from streamlit import streamlit as st   
from dotenv import load_dotenv
import os

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)


st.header("Reasearch Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
    template= """
        \nPlease summarize the research paper titled \"{paper_input}\" with the following specifications:\nExplanation Style: {style_input}  \nExplanation Length: {length_input}  \n1. Mathematical Details:  \n   - Include relevant mathematical equations if present in the paper.  \n   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  \n2. Analogies:  \n   - Use relatable analogies to simplify complex ideas.  \nIf certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.  \nEnsure the summary is clear, accurate, and aligned with the provided style and length.\n
    """,
    input_variables=["paper_input", "style_input", "length_input"]
)
# load_prompt('template.json')

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    result = chat_model.invoke(prompt)
    st.write(result.content)

# st.text_input("Enter your question here:", key="user_input")




# if st.button("Submit"):
#     user_input = st.session_state.user_input
#     response = chat_model.invoke(user_input)
#     st.write(f"User Input: {user_input}")
#     st.write(f"Response: {response.content}")



# # response = chat_model.invoke("What is the capital of India?")

# # print(response.content)