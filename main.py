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

# --- THE MINDMESH PERSONA ---
SYSTEM_PROMPT = """You are MindMesh, an elite AI study assistant. 
Your goal is to explain concepts clearly, intelligently, and empathetically.
Follow these rules strictly:
1. TONE: Be encouraging, highly analytical, and direct. 
2. FORMATTING: Use Markdown extensively. Use **bolding** for key terms, bullet points for lists, and headings (###) for structure.
3. TEACHING: Don't just give the final answer. Break down complex topics into simple, digestible steps.
4. CLOSING: Always end your response with a single, relevant follow-up question to keep the student engaged."""

# Setup Gemini 
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Setup Groq 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class StudyRequest(BaseModel):
    question: str
    is_complex: bool = False

@app.post("/ask")
async def ask_ai(request: StudyRequest):
    try:
        # 1. TRY GROQ FIRST (Llama 3.1)
        if not request.is_complex and GROQ_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama-3.1-8b-instant", 
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request.question}
                    ]
                }
                groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                groq_data = groq_response.json()
                
                if "choices" in groq_data:
                    answer = groq_data["choices"][0]["message"]["content"]
                    return {"answer": answer, "node": "⚡ Llama 3.1 Fast Node"}
            except Exception:
                pass # If Groq fails, silently pass to Gemini

        # 2. THE GEMINI SAFETY NET (Fallback & Complex queries)
        if gemini_api_key:
            full_prompt = f"{SYSTEM_PROMPT}\n\nStudent's query: {request.question}"
            response = gemini_model.generate_content(full_prompt)
            return {"answer": response.text, "node": "🤖 Gemini Deep-Think Node"}
            
        return {"answer": "⚠️ System offline. API keys missing.", "node": "Error"}

    except Exception as e:
        return {"answer": f"Critical System Error: {str(e)}", "node": "Error"}
