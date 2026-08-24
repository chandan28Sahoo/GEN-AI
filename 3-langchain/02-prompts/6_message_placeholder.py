from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []
history_file = Path(__file__).parent / 'chat_history.txt'

# load chat history
if history_file.exists():
    with open(history_file, 'r', encoding='utf-8') as f:
        chat_history.extend(f.readlines())

print("Loaded History:", chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

print("\nRendered Prompt:\n", prompt)