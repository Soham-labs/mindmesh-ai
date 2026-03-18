from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import requests 
import json
import PyPDF2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE ZYQORATH PERSONA ---
SYSTEM_PROMPT = """You are Zyqorath, an elite AI study assistant. 
Your goal is explain concepts clearly and intelligently.
1. TONE: Be encouraging, highly analytical, and direct. 
2. FORMATTING: Use Markdown extensively.
3. TEACHING: Break down complex topics into simple steps.
4. MEMORY: You have access to the conversation history. Refer back to it if the user asks follow-up questions."""

# Setup Gemini (Fallback & Heavy Lifter for PDFs)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro')
# Setup Groq (Fast Node)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@app.post("/ask")
async def ask_ai(
    question: str = Form(...),
    history: str = Form("[]"), # Receives the chat memory from frontend
    file: UploadFile = File(None) # Receives optional PDF
):
    try:
        # 1. Parse memory and set routing rules
        chat_history = json.loads(history)
        is_complex = len(question) > 300
        
        # 2. Handle PDF Uploads (RAG)
        if file and file.filename.endswith('.pdf'):
            is_complex = True # Force Gemini for large documents
            pdf_bytes = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            pdf_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text: pdf_text += text + "\n"
            
            # Combine PDF text with the user's question (capped at 30k chars for safety)
            question = f"Context from attached document:\n{pdf_text[:30000]}\n\nUser Question based on document: {question}"

        # 3. TRY GROQ FIRST (Llama 3.1)
        if not is_complex and GROQ_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                # Groq natively supports our memory array
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": question}]
                data = {"model": "llama-3.1-8b-instant", "messages": messages}
                
                groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                groq_data = groq_response.json()
                
                if "choices" in groq_data:
                    answer = groq_data["choices"][0]["message"]["content"]
                    return {"answer": answer, "node": "⚡ Llama 3.1 Fast Node"}
            except Exception:
                pass # Silently fallback to Gemini

        # 4. THE GEMINI SAFETY NET (Also handles all PDF reading)
        if gemini_api_key:
            # Convert memory array into a text transcript for Gemini
            transcript = f"{SYSTEM_PROMPT}\n\n"
            for msg in chat_history:
                role = "Student" if msg["role"] == "user" else "Zyqorath"
                transcript += f"{role}: {msg['content']}\n\n"
            transcript += f"Student: {question}\n\nZyqorath:"
            
            response = gemini_model.generate_content(transcript)
            return {"answer": response.text, "node": "🤖 Gemini Deep-Think Node"}
            
        return {"answer": "⚠️ System offline. API keys missing.", "node": "Error"}

    except Exception as e:
        return {"answer": f"Critical System Error: {str(e)}", "node": "Error"}
