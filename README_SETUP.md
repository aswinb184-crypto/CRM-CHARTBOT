# CRM Document Intelligence Chatbot

## Requirements

- Windows
- Python 3.12
- The project virtual environment in `.venv`
- A Groq API key in the local `.env` file for AI-generated chat answers

The Flask API and Streamlit interface bind to `127.0.0.1`; they are intended for
local access on this computer.

## Setup

Open PowerShell in the project folder:

```powershell
Set-Location 'D:\Downlods\CRM_Chatbot_RAG_Portable_Download\CRM_Chatbot_RAG_Portable'
```

If `.venv` does not exist, run the dependency setup once:

```powershell
.\setup.bat
```

If `.env` does not exist, create it from the example and add your own key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Keep `.env` private. Do not copy it into a ZIP or share its contents.

## Start the application

In the first PowerShell window, start the Flask backend:

```powershell
Set-Location 'D:\Downlods\CRM_Chatbot_RAG_Portable_Download\CRM_Chatbot_RAG_Portable'
.\run_backend.bat
```

In a second PowerShell window, start Streamlit:

```powershell
Set-Location 'D:\Downlods\CRM_Chatbot_RAG_Portable_Download\CRM_Chatbot_RAG_Portable'
.\run_streamlit.bat
```

Keep both windows open. Open `http://127.0.0.1:8501` in your browser. Stop each
server with **Ctrl+C** in its own window.

The chat input accepts PDF, DOCX, TXT, PNG, JPG, JPEG, and WEBP files, with a
20 MB per-file limit. Uploaded files are processed by the local Flask backend
and indexed for retrieval. Sending a file without a question asks the assistant
for a summary. The Flask API also enforces its own file-type, size, and content
validation.

Assistant responses include a **Read aloud** control, and the chat input area
includes **Start Voice**, **Stop Voice**, and **Stop Speech** controls. Text to
speech uses the browser's built-in speech synthesis; voice input uses its Web
Speech API speech-recognition implementation and only starts after **Start Voice**
is clicked. Voice recognition support varies by browser. Current
Chrome/Edge builds generally support it; other browsers may report it as
unsupported. Microphone access may require permission, and some browser
recognition implementations process audio using the browser provider's service.
Text-to-speech uses only voices the browser marks as local; if none are
available, the response is not sent to an external speech service.

## Verify the installation

Run these commands from the project root:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py streamlit_app.py security.py voice_input.py
.\.venv\Scripts\python.exe -m pytest --import-mode=importlib tests -q
```

The backend health check is available at `http://127.0.0.1:5001/health`. If
Streamlit reports that port 8501 is already in use, stop the other local
Streamlit process before starting this project.
