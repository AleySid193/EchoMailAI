import streamlit as st
import whisper
import google.generativeai as genai
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="EchoMailAI", page_icon="\ud83d\udce7")
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Initialize session state variables
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "email_content" not in st.session_state:
    st.session_state.email_content = None

# Function to reset session state (discard recording)
def reset_state():
    st.session_state.transcribed_text = None
    st.session_state.email_content = None
    st.rerun()  # Refresh UI

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
        return f"\u274c Error: {str(e)}"

# Function to open Gmail with the generated email
def open_gmail(subject, body):
    formatted_body = body.replace("\n", "%0A")  # Convert new lines to URL-encoded format
    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&su={subject}&body={formatted_body}"
    st.markdown(f"[Open in Gmail]({gmail_url})", unsafe_allow_html=True)

# Streamlit UI
st.title("\ud83c\udfa7 Welcome to EchoMail AI!")
st.write("Record your voice, transcribe it, and generate an email!")

# Record Audio
wav_audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="mic")

if wav_audio:
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.write(wav_audio)
    temp_wav.close()
    
    text = transcribe_audio(temp_wav.name)
    st.session_state.transcribed_text = text
    st.markdown(f"\ud83d\udcdd **Transcribed Text:**  \n{text}")

# Generate Email Button
if st.session_state.transcribed_text and not st.session_state.email_content:
    if st.button("Generate Email"):
        email_content = generate_email_content(st.session_state.transcribed_text)
        st.session_state.email_content = email_content
        st.rerun()  # Refresh UI

# Show email and options
if st.session_state.email_content:
    st.write("\ud83d\udce7 **Generated Email:**", st.session_state.email_content)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open in Gmail"):
            open_gmail("Generated Email", st.session_state.email_content)
    with col2:
        if st.button("\u274c Discard & Restart"):
            reset_state()
