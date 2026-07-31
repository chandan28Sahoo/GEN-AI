from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser
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

# JSON Schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review."
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review."
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return either pos or neg."
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"}
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"}
        },
        "name": {
            "type": ["string", "null"]
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

# Output Parser
parser = JsonOutputParser()

review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos.

The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches.

The 200MP camera is stunning, especially in night mode.

However, the phone is heavy, One UI contains bloatware, and the $1,300 price is very high.

Review by Chandan Sahoo
"""

prompt = f"""
Extract the information from the following review.

Return ONLY a valid JSON object that strictly follows this schema.

Schema:
{json_schema}

{parser.get_format_instructions()}

Review:
{review}
"""

response = chat_model.invoke(prompt)

print("Raw Response:\n")
print(response.content)

# Remove markdown if present
text = response.content.strip()

if text.startswith("```"):
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```").strip()

# Parse JSON
result = parser.parse(text)

print("\nParsed Output:\n")
print("Summary:", result["summary"])
print("Themes:", result["key_themes"])
print("Sentiment:", result["sentiment"])
print("Pros:", result.get("pros"))
print("Cons:", result.get("cons"))
print("Reviewer:", result.get("name"))