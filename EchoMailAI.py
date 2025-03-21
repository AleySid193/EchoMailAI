import streamlit as st
import whisper
import google.generativeai as genai
import tempfile
import os
from streamlit_mic_recorder import mic_recorder
import urllib.parse

st.set_page_config(page_title="EchoMailAI", page_icon="📧")
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
    st.rerun()  # Force a rerun of the app to refresh the UI

# Function to transcribe using Whisper
def transcribe_audio(filename):
    model = whisper.load_model("small")
    result = model.transcribe(filename, language="en")
    return result["text"]

# Function to generate an email with Gemini AI
def generate_email_content(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Write a professional email based on this text: {prompt}")
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to open Gmail with the generated email
def open_gmail(subject, body):
    encoded_subject = urllib.parse.quote(subject)
    encoded_body = urllib.parse.quote(body)
    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&su={encoded_subject}&body={encoded_body}"
    st.markdown(f'<a href="{gmail_url}" target="_blank">Open in Gmail</a>', unsafe_allow_html=True)

# Streamlit UI
st.title("Welcome to EchoMail AI!")
st.write("Record your voice, transcribe it, and generate an email!")

# Record Audio
wav_audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="mic")

if wav_audio:
    if isinstance(wav_audio, dict) and "bytes" in wav_audio:  # ✅ Extract raw bytes
        audio_bytes = wav_audio["bytes"]

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.write(audio_bytes)  # ✅ Write actual bytes
        temp_wav.close()

        text = transcribe_audio(temp_wav.name)
        st.session_state.transcribed_text = text
        st.markdown(f"**Transcribed Text:**  \n{text}")
    else:
        st.error("❌ Error: No valid audio recorded.")

# Generate Email Button
if st.session_state.transcribed_text and not st.session_state.email_content:
    if st.button("Generate Email"):
        email_content = generate_email_content(st.session_state.transcribed_text)
        st.session_state.email_content = email_content

# Show email and options
if st.session_state.email_content:
    st.write(" **Generated Email:**", st.session_state.email_content)
    
    col1, col2 = st.columns(2)
    with col1:
        open_gmail("Generated Email", st.session_state.email_content)  # Display the Gmail link
    with col2:
        if st.button("Discard & Restart"):
            reset_state()
