from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import requests 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Gemini (Our heavy-lifter and safety net)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Setup Groq (Our fast tutor)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class StudyRequest(BaseModel):
    question: str
    is_complex: bool = False

@app.post("/ask")
async def ask_ai(request: StudyRequest):
    try:
        # 1. TRY GROQ FIRST (If it's a short question)
        if not request.is_complex and GROQ_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama3-8b-8192", 
                    "messages": [{"role": "user", "content": f"You are a helpful study tutor. Answer this: {request.question}"}]
                }
                
                groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                groq_data = groq_response.json()
                
                # If Groq successfully gives an answer, send it!
                if "choices" in groq_data:
                    answer = groq_data["choices"][0]["message"]["content"]
                    return {"answer": f"<strong>⚡ [Llama 3 Node]:</strong><br><br>{answer.replace('\n', '<br>')}"}
            except Exception:
                # If Groq fails for ANY reason, we ignore the error and move to Gemini
                pass 

        # 2. THE GEMINI SAFETY NET (Used for long text, or if Groq crashes)
        if gemini_api_key:
            response = gemini_model.generate_content(
                f"You are an expert AI tutor. Please help the user with this: {request.question}"
            )
            return {"answer": f"<strong>🤖 [Gemini Node]:</strong><br><br>{response.text.replace('\n', '<br>')}"}
            
        # 3. IF BOTH FAIL (Only happens if API keys are missing in Render)
        return {"answer": "⚠️ System offline. Please check that your API keys are saved in Render."}

    except Exception as e:
        return {"answer": f"Critical System Error: {str(e)}"}
