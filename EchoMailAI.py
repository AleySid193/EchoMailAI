import streamlit as st
import whisper
import google.generativeai as genai
import sounddevice as sd
import numpy as np
import wave
import webbrowser
import tempfile
import time
import os

st.set_page_config(page_title="EchoMailAI", page_icon="📧")
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Audio Recording Settings
SAMPLE_RATE = 16000

# Initialize session state variables
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "email_content" not in st.session_state:
    st.session_state.email_content = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "transcription_done" not in st.session_state:
    st.session_state.transcription_done = False

# Function to reset session state (discard recording)
def reset_state():
    st.session_state.recording = False
    st.session_state.audio_data = None
    st.session_state.transcribed_text = None
    st.session_state.email_content = None
    st.session_state.filename = None
    st.session_state.transcription_done = False
    st.rerun()  # Refresh UI

# Function to record audio
def record_audio():
    st.session_state.recording = True
    st.write("🎤 Recording... Click 'Stop' to finish.")
    duration = 5  # Set duration
    st.session_state.audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()  # Ensure recording finishes

# Function to stop recording and save the file
def stop_recording():
    audio_data = st.session_state.get("audio_data")

    if audio_data is None or len(audio_data) == 0:
        st.warning("⚠️ No recording found! Please start recording first.")
        return None

    st.session_state.recording = False
    time.sleep(0.5)  # Ensure last frames are captured
    sd.stop()
    st.success("✅ Recording stopped!")

    # Convert to NumPy array if not already
    audio_data = np.array(audio_data, dtype=np.int16)

    # Save to temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())  # Ensure full audio is saved

    st.session_state.filename = temp_wav.name  # Store filename
    return temp_wav.name

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

# ✅ Hide "Start Recording" and "Discard" after email is generated
if not st.session_state.email_content:
    if st.button("Start Recording"):
        record_audio()

    if st.session_state.recording and st.button("Stop Recording"):
        filename = stop_recording()
        if filename:
            text = transcribe_audio(filename)
            st.session_state.transcribed_text = text
            st.session_state.transcription_done = True
            st.markdown(f"📝 **Transcribed Text:**  \n{text}")

# ✅ Show discard option **before generating email**
if st.session_state.transcription_done and not st.session_state.email_content:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Discard Recording"):
            reset_state()  # Reset app state

    with col2:
        if st.button("Generate Email"):
            email_content = generate_email_content(st.session_state.transcribed_text)
            st.session_state.email_content = email_content
            st.rerun()  # Refresh UI

# ✅ Show email only after confirmation, hide previous buttons
if st.session_state.email_content:
    st.write("📧 Generated Email:", st.session_state.email_content)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open in Gmail"):
            open_gmail("Generated Email", st.session_state.email_content)

    with col2:
        if st.button("❌ Discard & Restart"):
            reset_state()
