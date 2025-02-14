import streamlit as st
import whisper
import google.generativeai as genai
import numpy as np
import av
import tempfile
import os
import webbrowser
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, AudioProcessorBase

# Set Streamlit Page Config
st.set_page_config(page_title="EchoMail AI", page_icon="📧")

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("⚠️ Missing GEMINI API Key. Set GEMINI_API_KEY in environment variables.")
    st.stop()  # Stop execution if API key is missing

genai.configure(api_key=API_KEY)

# Initialize session state variables
if "audio_data" not in st.session_state:
    st.session_state.audio_data = []
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "email_content" not in st.session_state:
    st.session_state.email_content = None
if "recording_complete" not in st.session_state:
    st.session_state.recording_complete = False

# Function to reset session state
def reset_state():
    st.session_state.audio_data = []
    st.session_state.transcribed_text = None
    st.session_state.email_content = None
    st.session_state.recording_complete = False
    st.rerun()  # Refresh UI

# Function to transcribe using Whisper
def transcribe_audio(filename):
    model = whisper.load_model("base")  # Use "base" for Streamlit Cloud efficiency
    result = model.transcribe(filename)
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

# 🎤 **WebRTC Audio Recorder**
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_array = frame.to_ndarray()
        self.audio_buffer.append(audio_array)
        st.session_state.audio_data.append(audio_array)  # Store in session state
        return frame

st.title("🎙️ Welcome to EchoMail AI!")
st.write("Record your voice, transcribe it, and generate an email!")

# ✅ WebRTC Audio Recording
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioProcessor,
)

# ✅ Process Recorded Audio
if st.session_state.audio_data and not st.session_state.recording_complete:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio_path = temp_audio.name
        temp_audio.close()

    # 🔹 Convert recorded audio to a valid WAV file
    audio_data_np = np.concatenate(st.session_state.audio_data, axis=0)  # Flatten buffer
    with av.open(temp_audio_path, "w") as output_file:
        frame = av.AudioFrame.from_ndarray(audio_data_np, format="s16")
        output_file.write(frame)

    # Transcribe using Whisper
    text = transcribe_audio(temp_audio_path)
    st.session_state.transcribed_text = text
    st.session_state.recording_complete = True
    st.markdown(f"📝 **Transcribed Text:**  \n{text}")

# ✅ Generate Email After Transcription
if st.session_state.transcribed_text and not st.session_state.email_content:
    if st.button("Generate Email"):
        st.session_state.email_content = generate_email_content(st.session_state.transcribed_text)
        st.rerun()  # Refresh UI

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
