import cv2
import os
import time
import tempfile
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import supervision as sv

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
st.set_page_config(page_title="Conteo Vehicular Global", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")

# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path, task="detect")

def save_temp_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def list_local_videos():
    if not os.path.exists(VIDEOS_DIR):
        return []
    return [
        v for v in os.listdir(VIDEOS_DIR)
        if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

# --------------------------------------------------
# CLASES (AJUSTA A TU MODELO)
# --------------------------------------------------
CLASS_MAPPING = {
    "car": "Vehículo liviano",
    "pickup": "Vehículo liviano",
    "van": "Vehículo liviano",
    "motorbike": "Vehículo liviano",
    "bus": "Bus",
    "truck": "Camión"
}

VEHICLE_CLASSES = list(CLASS_MAPPING.keys())

# --------------------------------------------------
# INTERFAZ
# --------------------------------------------------
st.title("🚗 Conteo Vehicular Global")
st.markdown("Cuenta todos los vehículos del video una sola vez")

col1, col2 = st.columns(2)

# MODELO
with col1:
    model_option = st.radio(
        "Modelo",
        ["Usar best.onnx", "Subir modelo"],
        horizontal=True
    )
    model_path = MODEL_PATH if model_option == "Usar best.onnx" else None
    uploaded_model = None

    if model_option == "Subir modelo":
        uploaded_model = st.file_uploader("Sube modelo ONNX", type=["onnx"])

# VIDEO / CÁMARA
with col2:
    source_option = st.radio(
        "Fuente",
        ["Video local", "Subir video", "Cámara"],
        horizontal=True
    )

    video_path = None
    uploaded_video = None
    camera_source = None

    if source_option == "Video local":
        videos = list_local_videos()
        if videos:
            selected = st.selectbox("Videos", videos)
            video_path = os.path.join(VIDEOS_DIR, selected)
        else:
            st.warning("No hay videos en la carpeta Videos/")
    elif source_option == "Subir video":
        uploaded_video = st.file_uploader(
            "Sube un video",
            type=["mp4", "avi", "mov", "mkv"]
        )
    else:
        camera_source = st.text_input("Cámara (0, 1 o RTSP)", value="0")

fps_limit = st.slider("FPS máximo", 5, 30, 15)

# --------------------------------------------------
# ESTADO
# --------------------------------------------------
if "counted_ids" not in st.session_state:
    st.session_state.counted_ids = set()
    st.session_state.total = 0
    st.session_state.count = {
        "Vehículo liviano": 0,
        "Bus": 0,
        "Camión": 0
    }

# --------------------------------------------------
# INICIAR
# --------------------------------------------------
if st.button("▶ Iniciar conteo"):

    st.session_state.counted_ids.clear()
    st.session_state.total = 0
    for k in st.session_state.count:
        st.session_state.count[k] = 0

    if uploaded_model:
        model_path = save_temp_file(uploaded_model)
    if uploaded_video:
        video_path = save_temp_file(uploaded_video)

    model = load_model(model_path)
    tracker = sv.ByteTrack()

    if source_option == "Cámara":
        cap = cv2.VideoCapture(
            int(camera_source) if camera_source.isdigit() else camera_source
        )
    else:
        cap = cv2.VideoCapture(video_path)

    video_col, stats_col = st.columns([3, 1])
    frame_box = video_col.empty()
    stats_box = stats_col.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.4,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for box, track_id, cls_id in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.class_id
        ):
            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            class_name = model.names[int(cls_id)]
            label = CLASS_MAPPING.get(class_name, class_name)

            # ---- CONTEO GLOBAL (UNA SOLA VEZ) ----
            if (
                track_id not in st.session_state.counted_ids
                and class_name in VEHICLE_CLASSES
            ):
                st.session_state.counted_ids.add(track_id)
                st.session_state.total += 1
                st.session_state.count[label] += 1

            # COLOR SEGÚN ESTADO
            if track_id in st.session_state.counted_ids:
                color = (0, 0, 255)  # ROJO
                status = "CONTADO"
            else:
                color = (0, 255, 0)
                status = ""

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            cv2.putText(
                frame,
                f"{label} ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if status:
                cv2.putText(
                    frame,
                    status,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        with stats_box.container():
            st.metric("🚗 Total", st.session_state.total)
            for k, v in st.session_state.count.items():
                st.metric(k, v)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_box.image(frame, use_container_width=True)

        time.sleep(1 / fps_limit)

    cap.release()

    df = pd.DataFrame({
        "Tipo": st.session_state.count.keys(),
        "Cantidad": st.session_state.count.values()
    })

    st.subheader("📊 Resultados finales")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Descargar CSV",
        df.to_csv(index=False),
        "conteo_vehicular_global.csv",
        "text/csv"
    )
