import streamlit as st
import whisper
import google.generativeai as genai
import os
import tempfile
from streamlit_mic_recorder import st_mic_recorder
import webbrowser

# Set Streamlit Page Config
st.set_page_config(page_title="EchoMail AI", page_icon="📧")

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Initialize session state variables
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "email_content" not in st.session_state:
    st.session_state.email_content = None

# Function to reset state
def reset_state():
    st.session_state.audio_data = None
    st.session_state.transcribed_text = None
    st.session_state.email_content = None
    st.rerun()

# Function to transcribe using Whisper
def transcribe_audio(filename):
    model = whisper.load_model("medium")
    result = model.transcribe(filename, language="en")
    return result["text"]

# Function to generate an email with Gemini AI
def generate_email_content(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Write a professional email based on this text: {prompt}")
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Function to open Gmail with the generated email
def open_gmail(subject, body):
    formatted_body = body.replace("\n", "%0A")  # Convert new lines to URL-encoded format
    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&su={subject}&body={formatted_body}"
    webbrowser.open(gmail_url)

# Streamlit UI
st.title("🎙️ Welcome to EchoMail AI!")
st.write("Record your voice, transcribe it, and generate an email!")

# 🎤 **Record Audio Using `st_mic_recorder`**
audio_bytes = st_mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="recorder")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")  # Playback recorded audio

    # Save audio to a temporary file
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.write(audio_bytes)
    temp_audio.close()
    
    # Transcribe audio using Whisper
    text = transcribe_audio(temp_audio.name)
    st.session_state.transcribed_text = text

    # Show Transcription
    st.markdown(f"📝 **Transcribed Text:**  \n{text}")

# ✅ Generate Email After Transcription
if st.session_state.transcribed_text and not st.session_state.email_content:
    if st.button("Generate Email"):
        st.session_state.email_content = generate_email_content(st.session_state.transcribed_text)
        st.rerun()

# ✅ Show Email Content
if st.session_state.email_content:
    st.write("📧 **Generated Email:**", st.session_state.email_content)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open in Gmail"):
            open_gmail("Generated Email", st.session_state.email_content)

    with col2:
        if st.button("❌ Discard & Restart"):
            reset_state()
