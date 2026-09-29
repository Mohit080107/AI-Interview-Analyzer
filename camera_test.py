import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("🎥 Webcam Test")


class CameraTest(VideoProcessorBase):

    def recv(self, frame):
        return frame


webrtc_streamer(
    key="camera-test",
    video_processor_factory=CameraTest,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=False
)