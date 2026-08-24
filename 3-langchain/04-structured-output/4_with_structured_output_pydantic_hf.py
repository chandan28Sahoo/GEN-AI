from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal, Optional
from pydantic import BaseModel, Field
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


# Schema
class ReviewOutput(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(
        description="A brief summary of the review"
    )
    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either pos or neg"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )
    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )


# Create parser
parser = PydanticOutputParser(pydantic_object=ReviewOutput)

review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor (great for gaming and productivity)
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful

Review by Chandan Sahoo
"""

# Prompt with format instructions
prompt = f"""
Extract the following information from the review.

{parser.get_format_instructions()}

Review:
{review}
"""

# Invoke model
response = chat_model.invoke(prompt)

# Print raw output
print("Raw Response:\n")
print(response.content)

# Parse response
result = parser.parse(response.content)

print("\nParsed Output:\n")
print("Summary:", result.summary)
print("Themes:", result.key_themes)
print("Sentiment:", result.sentiment)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Reviewer:", result.name)