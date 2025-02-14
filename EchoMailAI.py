import streamlit as st
import whisper
import google.generativeai as genai
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import wave

st.set_page_config(page_title="EchoMail AI", page_icon="📧")

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

SAMPLE_RATE = 16000

# Session State Initialization
if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "email_content" not in st.session_state:
    st.session_state.email_content = None

# WebRTC Audio Recorder
st.title("🎙️ EchoMail AI - Record & Transcribe Audio")

def audio_callback(frame: av.AudioFrame):
    """ Processes audio frames from WebRTC """
    audio = frame.to_ndarray()
    return av.AudioFrame.from_ndarray(audio, layout="mono")

webrtc_ctx = webrtc_streamer(
    key="record-audio",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    ),
    audio_receiver_size=1024,
)

# Function to save WebRTC audio to a file
def save_audio():
    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            audio_data = np.concatenate([frame.to_ndarray() for frame in audio_frames])
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            with wave.open(temp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())

            st.session_state.audio_filename = temp_wav.name
            return temp_wav.name
    return None

# Button to Save & Transcribe Audio
if st.button("Stop & Transcribe"):
    filename = save_audio()
    if filename:
        model = whisper.load_model("base")
        text = model.transcribe(filename)["text"]
        st.session_state.transcribed_text = text
        st.success("✅ Audio transcribed successfully!")

# Display Transcribed Text
if st.session_state.transcribed_text:
    st.write("📝 **Transcribed Text:**", st.session_state.transcribed_text)

# Generate Email with Gemini AI
if st.session_state.transcribed_text and st.button("Generate Email"):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Write a professional email based on this text: {st.session_state.transcribed_text}")
        st.session_state.email_content = response.text
        st.success("📧 Email generated successfully!")
    except Exception as e:
        st.error(f"❌ Error generating email: {str(e)}")

# Display Generated Email
if st.session_state.email_content:
    st.write("📧 **Generated Email:**", st.session_state.email_content)
