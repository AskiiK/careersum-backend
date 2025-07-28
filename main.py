# main.py (Version for Google Gemini - Corrected)
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi.responses import StreamingResponse
import asyncio

# --- Google Sheets Logging Setup ---
def log_to_sheet(user_question, ai_response):
    try:
        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        
        # Use the JSON file for credentials
        creds = ServiceAccountCredentials.from_json_keyfile_name("google_credentials.json", scope )
        client = gspread.authorize(creds)
        
        # Open the sheet by its name
        sheet = client.open("CareerSum Chat Logs").sheet1
        
        # Prepare the row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, user_question, ai_response]
        
        # Append the row to the sheet
        sheet.append_row(row)
        print("Successfully logged conversation to Google Sheet.")
    except Exception as e:
        print(f"Error logging to Google Sheet: {e}")
# --- End of Logging Setup ---


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
You are "CareerSum AI," a friendly, encouraging, and insightful career assistant...
... (keep all the existing personality and instructions) ...

Your services include:
**For Professionals:**
- 1:1 Career Strategy
- Resume & LinkedIn Makeovers
- Interview & Mock Sessions
- Return-to-Work Programs
- Corporate Workshops

**For Students & Graduates:**
- Stream Selection Guidance for Class 10 students.
- College & Major choice advisory for Class 12 students.
- Guidance for graduates on choosing between a first job and an MBA.

When the user is ready to take the next step, your primary call to action is to have them book a free discovery call...

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
        # --- Step 1: Call the Google AI API ---
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([CAREER_AGENT_PROMPT, request.message])
        ai_response = response.text

        # --- Step 2: If successful, log the conversation ---
        # This now only runs if the AI call succeeds.
        log_to_sheet(request.message, ai_response)

        # --- Step 3: Return the successful response ---
        return {"response": ai_response}

    except Exception as e:
        # --- This block now handles ALL errors ---
        error_message = f"Error calling Google AI: {e}"
        print(error_message)
        
        # Log the error itself to the sheet so you know it happened
        log_to_sheet(request.message, f"SERVER_ERROR: {error_message}")

        # Return a user-friendly error message
        return {"response": "Sorry, I'm having trouble thinking right now. Please try again in a moment."}


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
