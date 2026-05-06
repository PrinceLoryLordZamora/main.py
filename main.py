import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# ---------- PAGE ----------
st.set_page_config(page_title="Cyber AI Detection", layout="wide")

# ---------- CYBER UI STYLE ----------
st.markdown("""
<style>
/* Background */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
}

/* Neon Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
    text-shadow: 0 0 10px #38bdf8, 0 0 20px #0ea5e9;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Neon Card */
.card {
    background: rgba(2, 6, 23, 0.7);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(56,189,248,0.3);
    box-shadow: 0 0 15px rgba(56,189,248,0.2);
}

/* Neon Button */
.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
}

/* Slider text */
.css-1cpxqw2 {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">⚡ CYBER AI DETECTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Object Detection System</div>', unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns([3, 1])

# ---------- SETTINGS ----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🎛 Controls")

    confidence = st.slider("Confidence", 0.1, 0.9, 0.25)
    frame_skip = st.slider("Smoothness", 1, 5, 3)

    st.markdown("### ⚡ System Status")
    st.success("🟢 Running")
    st.caption("Optimized for smooth detection")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- PROCESSOR ----------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_result = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (480, 360))

        self.frame_count += 1

        if self.frame_count % frame_skip == 0:
            results = model.predict(img, conf=confidence, verbose=False)
            self.last_result = results
        else:
            results = self.last_result

        annotated = img.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = model.names

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])

                # neon box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,255,0), 2)

                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------- VIDEO ----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    webrtc_streamer(
        key="cyber-detect",
        video_processor_factory=VideoProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": 480,
                "height": 360,
                "frameRate": 15
            },
            "audio": False,
        },
    )

    st.markdown('</div>', unsafe_allow_html=True)