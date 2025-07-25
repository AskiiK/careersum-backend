# main.py (Version for Google Gemini - Corrected)

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi.responses import StreamingResponse
import asyncio


# Load environment variables from a .env file
load_dotenv()

# --- IMPORTANT: Configure the Google AI API key ---
# This happens once when the server starts.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable not set. The API will not work.")
else:
    genai.configure(api_key=api_key)
    print("Google AI SDK configured successfully.")

app = FastAPI()

# --- Allow connections from any frontend for now ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System Prompt for the AI ---
CAREER_AGENT_PROMPT = """
You are "CareerSum AI," a friendly, encouraging, and insightful career assistant for the career consulting service, CareerSum.
Your goal is to provide helpful, initial career advice and guide users to book a free discovery call with the human experts, Abhishek and Sanskriti.

Your personality:
- Professional but approachable.
- Positive and empowering.
- Knowledgeable about the tech industry, career changes, resumes, and interview prep.

Your instructions:
1. Keep your answers concise and easy to read (use bullet points where helpful).
2. Do NOT give financial or legal advice. Steer the conversation back to career strategy.
3. When the user is ready to take the next step, your primary call to action is to have them book a free discovery call.
4. To book a call, direct the user to this scheduling page: [Link](https://calendar.app.google/NumWkNh1rxgGQzAq6).
5. Do not make up links or use placeholders like [Insert Link Here]. Only use the provided email link.
"""

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        # Initialize the Gemini Pro model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Start a chat session, including the system prompt as the first message
        chat = model.start_chat(history=[
            {'role': 'user', 'parts': [CAREER_AGENT_PROMPT]},
            {'role': 'model', 'parts': ["Understood. I am CareerSum AI, ready to assist with career questions and guide users to book a discovery call."]}
        ])

        # Send the user's message to the chat session
        response = chat.send_message(request.message)
        
        ai_message = response.text
        return {"response": ai_message}
    except Exception as e:
        # This will catch errors if the API key was not configured at startup
        print(f"Error calling Google AI: {e}")
        return {"response": f"Sorry, an error occurred on the server. Check server logs for details."}

@app.get("/")
def read_root():
    return {"status": "CareerSum AI agent (Gemini Edition) is running."}

# The /chat endpoint would be rewritten to be an async generator
async def stream_chat_response(message: str):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    # Use stream=True to get chunks
    response_stream = await model.generate_content_async(message, stream=True) 
    async for chunk in response_stream:
        if chunk.text:
            yield f"data: {chunk.text}\n\n" # Format for Server-Sent Events
            await asyncio.sleep(0.01) # Small delay

@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(stream_chat_response(request.message), media_type="text/event-stream")
