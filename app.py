import streamlit as st
import cv2
import threading
import speech_recognition as sr
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎤",
    layout="wide"
)


# ============================================================
# PERSISTENT STATISTICS
# ============================================================

@st.cache_resource
def get_stats():

    return {
        "frames": 0,
        "face_frames": 0,
        "eye_frames": 0,
        "good_eye_contact": 0,
        "centered_head": 0,
        "score_total": 0,
        "lock": threading.Lock()
    }


stats = get_stats()


# ============================================================
# SESSION STATE
# ============================================================

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "word_count" not in st.session_state:
    st.session_state.word_count = 0

if "filler_count" not in st.session_state:
    st.session_state.filler_count = 0

if "voice_score" not in st.session_state:
    st.session_state.voice_score = 0

if "report_generated" not in st.session_state:
    st.session_state.report_generated = False


# ============================================================
# HEADER
# ============================================================

st.title("🎤 AI Interview Analyzer")

st.caption(
    "Practice interviews with AI-powered visual and voice feedback"
)

st.divider()


# ============================================================
# QUESTION
# ============================================================

question = st.selectbox(
    "Choose an Interview Question",
    [
        "Tell me about yourself",
        "What are your strengths?",
        "Why should we hire you?",
        "Describe a challenging project you worked on."
    ]
)

st.subheader("Your Interview Question")
st.success(question)

st.divider()


# ============================================================
# LIVE CAMERA
# ============================================================

st.subheader("🎥 Live Interview Analysis")


class InterviewAnalyzer(VideoProcessorBase):

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_eye.xml"
        )

        self.frame_count = 0

        # Important for smooth performance
        self.process_every = 15

        self.last_faces = []
        self.last_eyes = []

        self.eye_contact = "ANALYZING"
        self.head_position = "ANALYZING"

        self.score = 0


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Low resolution keeps webcam smooth
        image = cv2.resize(
            image,
            (360, 202),
            interpolation=cv2.INTER_LINEAR
        )

        self.frame_count += 1


        # ========================================================
        # DETECTION
        # ========================================================

        if self.frame_count % self.process_every == 0:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # Smaller image for faster face detection
            small_gray = cv2.resize(
                gray,
                (180, 101),
                interpolation=cv2.INTER_LINEAR
            )

            faces = self.face_cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )

            self.last_faces = []

            for (x, y, w, h) in faces:

                self.last_faces.append(
                    (
                        x * 2,
                        y * 2,
                        w * 2,
                        h * 2
                    )
                )

            self.last_eyes = []

            self.eye_contact = "NO FACE"
            self.head_position = "NOT DETECTED"
            self.score = 0


            # ====================================================
            # FACE FOUND
            # ====================================================

            if len(self.last_faces) > 0:

                x, y, w, h = max(
                    self.last_faces,
                    key=lambda f: f[2] * f[3]
                )


                # ------------------------------------------------
                # HEAD POSITION
                # ------------------------------------------------

                face_center_x = x + (w / 2)

                frame_center_x = (
                    image.shape[1] / 2
                )

                difference = (
                    face_center_x -
                    frame_center_x
                )

                ratio = (
                    difference /
                    image.shape[1]
                )

                if abs(ratio) < 0.15:

                    self.head_position = "CENTER"

                elif ratio < 0:

                    self.head_position = "LEFT"

                else:

                    self.head_position = "RIGHT"


                # ------------------------------------------------
                # EYE REGION
                # ------------------------------------------------

                eye_top = max(0, y)

                eye_bottom = min(
                    gray.shape[0],
                    y + int(h * 0.55)
                )

                eye_left = max(0, x)

                eye_right = min(
                    gray.shape[1],
                    x + w
                )


                if (
                    eye_bottom > eye_top
                    and
                    eye_right > eye_left
                ):

                    eye_region = gray[
                        eye_top:eye_bottom,
                        eye_left:eye_right
                    ]

                    eyes = self.eye_cascade.detectMultiScale(
                        eye_region,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(10, 10)
                    )

                    self.last_eyes = list(eyes)[:2]


                # ------------------------------------------------
                # EYE CONTACT
                # ------------------------------------------------

                if len(self.last_eyes) >= 1:

                    eye_centers = []

                    for (
                        ex,
                        ey,
                        ew,
                        eh
                    ) in self.last_eyes:

                        center_x = (
                            x +
                            ex +
                            ew / 2
                        )

                        eye_centers.append(
                            center_x
                        )

                    average_eye_x = (
                        sum(eye_centers)
                        /
                        len(eye_centers)
                    )

                    eye_difference = (
                        average_eye_x -
                        face_center_x
                    )

                    eye_ratio = (
                        abs(eye_difference)
                        /
                        max(w, 1)
                    )


                    if eye_ratio < 0.18:

                        self.eye_contact = "GOOD"

                    elif eye_ratio < 0.30:

                        self.eye_contact = (
                            "SLIGHTLY AWAY"
                        )

                    else:

                        self.eye_contact = (
                            "LOOKING AWAY"
                        )

                else:

                    self.eye_contact = (
                        "EYES NOT CLEAR"
                    )


                # ------------------------------------------------
                # VISUAL SCORE
                # ------------------------------------------------

                score = 40

                if len(self.last_eyes) >= 1:
                    score += 20

                if self.eye_contact == "GOOD":
                    score += 25

                elif self.eye_contact == "SLIGHTLY AWAY":
                    score += 15

                if self.head_position == "CENTER":
                    score += 15

                elif self.head_position in [
                    "LEFT",
                    "RIGHT"
                ]:
                    score += 5

                self.score = min(score, 100)


            # ====================================================
            # SAVE STATISTICS
            # ====================================================

            with stats["lock"]:

                stats["frames"] += 1

                if len(self.last_faces) > 0:
                    stats["face_frames"] += 1

                if len(self.last_eyes) >= 1:
                    stats["eye_frames"] += 1

                if self.eye_contact == "GOOD":
                    stats["good_eye_contact"] += 1

                if self.head_position == "CENTER":
                    stats["centered_head"] += 1

                stats["score_total"] += self.score


        # ========================================================
        # DRAW FACE
        # ========================================================

        for (
            x,
            y,
            w,
            h
        ) in self.last_faces:

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


        # ========================================================
        # STATUS
        # ========================================================

        cv2.putText(
            image,
            "FACE: " +
            (
                "DETECTED"
                if len(self.last_faces) > 0
                else "NOT DETECTED"
            ),
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0)
            if len(self.last_faces) > 0
            else (0, 0, 255),
            1
        )


        cv2.putText(
            image,
            "EYES: " +
            (
                "DETECTED"
                if len(self.last_eyes) >= 1
                else "NOT CLEAR"
            ),
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1
        )


        cv2.putText(
            image,
            "CONTACT: " +
            self.eye_contact,
            (10, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )


        cv2.putText(
            image,
            "HEAD: " +
            self.head_position,
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1
        )


        cv2.putText(
            image,
            "SCORE: " +
            str(self.score) +
            "/100",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )


        return frame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# START CAMERA
# ============================================================

webrtc_streamer(
    key="final-interview-camera",
    video_processor_factory=InterviewAnalyzer,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 360},
            "height": {"ideal": 202},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=False
)


# ============================================================
# VOICE ANALYSIS
# ============================================================

st.divider()

st.subheader("🎙️ Voice & Answer Analysis")

st.write(
    "Record your answer using your microphone."
)


audio_file = st.audio_input(
    "🎙️ Record Your Answer"
)


if audio_file is not None:

    st.audio(audio_file)

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(audio_file) as source:

            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(
            audio_data
        )

        st.session_state.transcript = transcript

        words = transcript.split()

        st.session_state.word_count = len(words)


        # ========================================================
        # FILLER WORDS
        # ========================================================

        filler_words = [
            "um",
            "uh",
            "like",
            "actually",
            "basically",
            "you know",
            "sort of",
            "kind of"
        ]

        transcript_lower = transcript.lower()

        filler_count = 0

        for filler in filler_words:

            filler_count += transcript_lower.count(
                filler
            )

        st.session_state.filler_count = filler_count


        # ========================================================
        # VOICE SCORE
        # ========================================================

        voice_score = 100

        if filler_count >= 8:

            voice_score -= 30

        elif filler_count >= 5:

            voice_score -= 20

        elif filler_count >= 3:

            voice_score -= 10


        if len(words) < 20:

            voice_score -= 25

        elif len(words) < 40:

            voice_score -= 10


        st.session_state.voice_score = max(
            0,
            min(
                voice_score,
                100
            )
        )


        # ========================================================
        # DISPLAY TRANSCRIPT
        # ========================================================

        st.success(
            "✅ Speech successfully converted!"
        )

        st.subheader("📝 Your Answer")

        st.write(transcript)


        # ========================================================
        # VOICE METRICS
        # ========================================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Words",
            len(words)
        )

        col2.metric(
            "Filler Words",
            filler_count
        )

        col3.metric(
            "Voice Score",
            f"{st.session_state.voice_score}/100"
        )


        # ========================================================
        # VOICE FEEDBACK
        # ========================================================

        st.subheader("💡 Voice Feedback")

        if len(words) < 20:

            st.warning(
                "Your answer is quite short. "
                "Try to provide more detail and examples."
            )

        elif len(words) >= 40:

            st.success(
                "Your answer contains a reasonable "
                "amount of detail."
            )

        else:

            st.info(
                "Your answer has moderate detail. "
                "Consider adding a specific example."
            )


        if filler_count >= 5:

            st.warning(
                "You used several filler words. "
                "Try using short pauses instead."
            )

        else:

            st.success(
                "Good control of filler words."
            )


    except sr.UnknownValueError:

        st.error(
            "❌ I could not understand the audio. "
            "Try recording again."
        )

    except sr.RequestError:

        st.error(
            "❌ Speech recognition service could not "
            "be reached. Check your internet connection."
        )

    except Exception as e:

        st.error(
            "Voice processing error: " +
            str(e)
        )


# ============================================================
# FINAL REPORT
# ============================================================

st.divider()

st.subheader("📊 Final Interview Report")

st.write(
    "Keep the webcam running for at least 30 seconds "
    "and record your answer before generating the report."
)


if st.button(
    "🏆 Generate Final Report"
):

    with stats["lock"]:

        total = stats["frames"]


        if total > 0:

            face_percentage = (
                stats["face_frames"]
                / total
                * 100
            )

            eye_percentage = (
                stats["eye_frames"]
                / total
                * 100
            )

            contact_percentage = (
                stats["good_eye_contact"]
                / total
                * 100
            )

            head_percentage = (
                stats["centered_head"]
                / total
                * 100
            )

            visual_score = (
                stats["score_total"]
                / total
            )

        else:

            face_percentage = 0
            eye_percentage = 0
            contact_percentage = 0
            head_percentage = 0
            visual_score = 0


    # ========================================================
    # ANSWER SCORE
    # ========================================================

    word_count = st.session_state.word_count

    filler_count = st.session_state.filler_count


    answer_score = 100


    if word_count < 20:

        answer_score -= 30

    elif word_count < 40:

        answer_score -= 10


    if filler_count >= 8:

        answer_score -= 25

    elif filler_count >= 5:

        answer_score -= 15

    elif filler_count >= 3:

        answer_score -= 5


    answer_score = max(
        0,
        min(
            answer_score,
            100
        )
    )


    # ========================================================
    # VOICE SCORE
    # ========================================================

    voice_score = st.session_state.voice_score


    # ========================================================
    # FINAL SCORE
    # ========================================================

    overall_score = (
        visual_score * 0.45
        +
        voice_score * 0.30
        +
        answer_score * 0.25
    )


    overall_score = max(
        0,
        min(
            overall_score,
            100
        )
    )


    # ========================================================
    # SCORE DISPLAY
    # ========================================================

    st.markdown("## 🏆 Overall Interview Score")

    st.progress(
        int(overall_score)
    )

    st.metric(
        "Overall Score",
        f"{overall_score:.1f} / 100"
    )


    st.divider()


    # ========================================================
    # CATEGORY SCORES
    # ========================================================

    st.subheader("📈 Performance Breakdown")

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "👁️ Visual",
        f"{visual_score:.1f}"
    )

    col2.metric(
        "🎙️ Voice",
        f"{voice_score:.1f}"
    )

    col3.metric(
        "📝 Answer",
        f"{answer_score:.1f}"
    )

    col4.metric(
        "👀 Eye Contact",
        f"{contact_percentage:.1f}%"
    )


    st.divider()


    # ========================================================
    # DETAILED METRICS
    # ========================================================

    st.subheader("📊 Detailed Analysis")

    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"👤 **Face Presence:** "
            f"{face_percentage:.1f}%"
        )

        st.write(
            f"👀 **Eye Detection:** "
            f"{eye_percentage:.1f}%"
        )

        st.write(
            f"🎯 **Eye Contact:** "
            f"{contact_percentage:.1f}%"
        )

        st.write(
            f"🧍 **Head Centered:** "
            f"{head_percentage:.1f}%"
        )


    with col2:

        st.write(
            f"📝 **Words Spoken:** "
            f"{word_count}"
        )

        st.write(
            f"🚫 **Filler Words:** "
            f"{filler_count}"
        )

        st.write(
            f"🎙️ **Voice Score:** "
            f"{voice_score}/100"
        )

        st.write(
            f"📝 **Answer Score:** "
            f"{answer_score}/100"
        )


    # ========================================================
    # STRENGTHS
    # ========================================================

    st.subheader("💪 Strengths")

    strengths = []


    if face_percentage >= 80:

        strengths.append(
            "Good camera presence."
        )


    if contact_percentage >= 60:

        strengths.append(
            "Maintained reasonable eye contact."
        )


    if head_percentage >= 70:

        strengths.append(
            "Head position was generally centered."
        )


    if filler_count < 3:

        strengths.append(
            "Low use of filler words."
        )


    if word_count >= 40:

        strengths.append(
            "Provided a reasonably detailed answer."
        )


    if not strengths:

        strengths.append(
            "Completed the interview practice session."
        )


    for strength in strengths:

        st.success(
            "✓ " + strength
        )


    # ========================================================
    # AREAS TO IMPROVE
    # ========================================================

    st.subheader("🎯 Areas to Improve")

    improvements = []


    if face_percentage < 80:

        improvements.append(
            "Stay clearly visible within the camera frame."
        )


    if contact_percentage < 60:

        improvements.append(
            "Try to maintain eye contact with the camera."
        )


    if head_percentage < 70:

        improvements.append(
            "Keep your head more centered."
        )


    if filler_count >= 3:

        improvements.append(
            "Reduce filler words such as 'um', 'uh', and 'like'."
        )


    if word_count < 40:

        improvements.append(
            "Give more detailed answers with specific examples."
        )


    if not improvements:

        improvements.append(
            "Continue practicing to maintain consistency."
        )


    for improvement in improvements:

        st.warning(
            "• " + improvement
        )


    # ========================================================
    # INTERVIEW TIP
    # ========================================================

    st.subheader("💡 Interview Tip")

    if word_count < 40:

        st.info(
            "For behavioral questions, try using the "
            "STAR method: Situation → Task → Action → Result."
        )

    elif filler_count >= 3:

        st.info(
            "Instead of using filler words, pause briefly "
            "before continuing your answer."
        )

    elif contact_percentage < 60:

        st.info(
            "Place the camera near eye level and look "
            "toward the camera while speaking."
        )

    else:

        st.info(
            "Keep practicing with different questions "
            "to improve consistency."
        )


    # ========================================================
    # STORE REPORT
    # ========================================================

    st.session_state.report_data = {
        "question": question,
        "overall_score": overall_score,
        "visual_score": visual_score,
        "voice_score": voice_score,
        "answer_score": answer_score,
        "face_percentage": face_percentage,
        "eye_percentage": eye_percentage,
        "contact_percentage": contact_percentage,
        "head_percentage": head_percentage,
        "word_count": word_count,
        "filler_count": filler_count,
        "transcript": st.session_state.transcript,
        "strengths": strengths,
        "improvements": improvements
    }


# ============================================================
# PDF REPORT
# ============================================================

if "report_data" in st.session_state:

    st.divider()

    st.subheader("📄 Download Interview Report")


    def create_pdf(data):

        filename = "AI_Interview_Report.pdf"

        c = canvas.Canvas(
            filename,
            pagesize=A4
        )

        width, height = A4

        y = height - 50


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            20
        )

        c.drawString(
            50,
            y,
            "AI Interview Analyzer"
        )

        y -= 30


        c.setFont(
            "Helvetica",
            10
        )

        c.drawString(
            50,
            y,
            "Interview Performance Report"
        )

        y -= 30


        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        c.drawString(
            50,
            y,
            "Generated: "
            +
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            )
        )

        y -= 30


        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            12
        )

        c.drawString(
            50,
            y,
            "Question:"
        )

        y -= 18


        c.setFont(
            "Helvetica",
            10
        )

        c.drawString(
            50,
            y,
            data["question"][:90]
        )

        y -= 30


        # ----------------------------------------------------
        # OVERALL SCORE
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            15
        )

        c.drawString(
            50,
            y,
            "Overall Score: "
            +
            f"{data['overall_score']:.1f}/100"
        )

        y -= 30


        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            12
        )

        c.drawString(
            50,
            y,
            "Performance Breakdown"
        )

        y -= 20


        c.setFont(
            "Helvetica",
            10
        )


        metrics = [
            (
                "Visual Score",
                data["visual_score"]
            ),
            (
                "Voice Score",
                data["voice_score"]
            ),
            (
                "Answer Score",
                data["answer_score"]
            ),
            (
                "Face Presence",
                data["face_percentage"]
            ),
            (
                "Eye Contact",
                data["contact_percentage"]
            ),
            (
                "Head Centered",
                data["head_percentage"]
            ),
            (
                "Words",
                data["word_count"]
            ),
            (
                "Filler Words",
                data["filler_count"]
            )
        ]


        for name, value in metrics:

            c.drawString(
                60,
                y,
                f"{name}: {value:.1f}"
                if isinstance(value, float)
                else f"{name}: {value}"
            )

            y -= 17


        y -= 10


        # ----------------------------------------------------
        # TRANSCRIPT
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            12
        )

        c.drawString(
            50,
            y,
            "Answer Transcript"
        )

        y -= 20


        c.setFont(
            "Helvetica",
            9
        )


        transcript = data[
            "transcript"
        ]


        # Wrap transcript
        words = transcript.split()

        line = ""


        for word in words:

            if len(line) + len(word) > 90:

                c.drawString(
                    50,
                    y,
                    line
                )

                y -= 14

                line = word

            else:

                line += " " + word


        if line:

            c.drawString(
                50,
                y,
                line
            )

            y -= 25


        # ----------------------------------------------------
        # STRENGTHS
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            12
        )

        c.drawString(
            50,
            y,
            "Strengths"
        )

        y -= 18


        c.setFont(
            "Helvetica",
            9
        )


        for strength in data["strengths"]:

            c.drawString(
                60,
                y,
                "- " + strength
            )

            y -= 15


        y -= 10


        # ----------------------------------------------------
        # IMPROVEMENTS
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Bold",
            12
        )

        c.drawString(
            50,
            y,
            "Areas to Improve"
        )

        y -= 18


        c.setFont(
            "Helvetica",
            9
        )


        for improvement in data[
            "improvements"
        ]:

            c.drawString(
                60,
                y,
                "- " + improvement
            )

            y -= 15


        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        c.setFont(
            "Helvetica-Oblique",
            8
        )

        c.drawString(
            50,
            30,
            "Prototype computer-vision and speech-analysis metrics."
        )


        c.save()

        return filename


    pdf_file = create_pdf(
        st.session_state.report_data
    )


    with open(
        pdf_file,
        "rb"
    ) as file:

        st.download_button(
            label="📄 Download PDF Report",
            data=file,
            file_name="AI_Interview_Report.pdf",
            mime="application/pdf"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Interview Analyzer | "
    "Computer Vision + Speech Recognition + Streamlit"
)