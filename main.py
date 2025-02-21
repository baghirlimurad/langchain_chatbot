import json 
import os

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

open_api_key = os.getenv("OPENAI_API_KEY")
if not open_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# Initialize FastAPI app
app = FastAPI()

# Define message model
class Message(BaseModel):
    text: str

# Define conversation prompt
conversation_prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template="""
    You are an AI Workshop assistant who helps with registrations and questions.
    
    Previous conversation:
    {history}
    
    Human: {human_input}
    AI Assistant: """
)

# Initialize ChatOpenAI and Memory
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="history")

# Create conversation chain
conversation_chain = LLMChain(
    llm=llm,
    prompt=conversation_prompt,
    memory=memory,
    verbose=True
)

# Example usage with custom knowledge from data.json
with open("data/data.json", "r") as f:
    WORKSHOP_INFO = json.load(f)

@app.post("/chat")
async def chat(message: Message):
    # Get response from AI
    response = conversation_chain.predict(human_input=message.text)
    return {"response": response}

# Function to inject workshop knowledge
def inject_workshop_knowledge():
    context = f"""
    This is a workshop about {WORKSHOP_INFO['title']}.
    It lasts {WORKSHOP_INFO['duration']} and has {WORKSHOP_INFO['max_participants']} spots available.
    Topics covered: {', '.join(WORKSHOP_INFO['topics'])}
    """
    conversation_chain.predict(human_input=f"Please remember this context: {context}")

# Initialize knowledge
@app.on_event("startup")
async def startup_event():
    inject_workshop_knowledge()