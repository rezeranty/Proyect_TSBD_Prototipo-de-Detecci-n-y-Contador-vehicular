import cv2
import os
import streamlit as st
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

# -----------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------
st.set_page_config(page_title="Conteo Vehicular", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")

# -----------------------------------------------------------
# CARGA DEL MODELO
# -----------------------------------------------------------
st.title("🚗 Conteo de Vehículos con Línea – YOLOv8 + ByteTrack")

if not os.path.exists(MODEL_PATH):
    st.error("❌ No se encontró el modelo best.onnx")
    st.stop()

model = YOLO(MODEL_PATH, task="detect")
st.success("Modelo cargado correctamente ✔")

# Tracker REAL (ByteTrack)
tracker = sv.ByteTrack()

# -----------------------------------------------------------
# CLASES DEL MODELO → CATEGORÍAS
# (AJUSTA ESTOS NOMBRES EXACTAMENTE A TU MODELO)
# -----------------------------------------------------------
CLASS_MAPPING = {
    "bus": "bus",
    "truck": "camion",
    "car": "vehiculo liviano",
    "pickup": "vehiculo liviano",
    "van": "vehiculo liviano"
}

# -----------------------------------------------------------
# LISTAR VIDEOS
# -----------------------------------------------------------
def listar_videos():
    if not os.path.exists(VIDEOS_DIR):
        return []
    return [v for v in os.listdir(VIDEOS_DIR)
            if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

videos = listar_videos()
if not videos:
    st.error("❌ No hay videos en la carpeta Videos/")
    st.stop()

video_sel = st.selectbox("🎬 Selecciona un video", videos)
VIDEO_PATH = os.path.join(VIDEOS_DIR, video_sel)

if st.button("📤 Mostrar video"):
    st.video(VIDEO_PATH)

# -----------------------------------------------------------
# LÍNEA DE CONTEO
# -----------------------------------------------------------
st.subheader("⚙ Configuración de línea de conteo")

line_pos = st.slider(
    "Posición vertical de la línea",
    min_value=50,
    max_value=700,
    value=250   # 🔥 ESTE VALOR ES EL CORRECTO PARA TU VIDEO
)

# -----------------------------------------------------------
# ESTADO GLOBAL (STREAMLIT)
# -----------------------------------------------------------
if "count_by_class" not in st.session_state:
    st.session_state.count_by_class = {
        "bus": 0,
        "camion": 0,
        "vehiculo liviano": 0
    }

if "counted_ids" not in st.session_state:
    st.session_state.counted_ids = set()

if "total_count" not in st.session_state:
    st.session_state.total_count = 0

# -----------------------------------------------------------
# PROCESAR VIDEO
# -----------------------------------------------------------
if st.button("▶ Iniciar Conteo"):

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_display = st.empty()

    class_names = model.names

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------
        # DETECCIÓN
        # -------------------------
        results = model(frame, conf=0.4)[0]
        detections = sv.Detections.from_ultralytics(results)

        # -------------------------
        # TRACKING
        # -------------------------
        detections = tracker.update_with_detections(detections)

        # -------------------------
        # DIBUJAR LÍNEA
        # -------------------------
        cv2.line(
            frame,
            (0, line_pos),
            (frame.shape[1], line_pos),
            (0, 255, 255),
            4
        )

        # -------------------------
        # PROCESAR TRACKS
        # -------------------------
        for box, track_id, cls_id in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.class_id
        ):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            class_name = class_names[int(cls_id)]

            if class_name not in CLASS_MAPPING:
                continue

            category = CLASS_MAPPING[class_name]

            # Dibujar bounding box GRANDE
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3
            )

            # -------------------------
            # CONTEO REAL
            # -------------------------
            if (
                track_id not in st.session_state.counted_ids
                and abs(cy - line_pos) < 20
            ):
                st.session_state.counted_ids.add(track_id)
                st.session_state.total_count += 1
                st.session_state.count_by_class[category] += 1

        # -------------------------
        # MOSTRAR CONTADORES EN VIVO
        # -------------------------
        cv2.putText(frame, f"Total: {st.session_state.total_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3)

        y = 90
        for k, v in st.session_state.count_by_class.items():
            cv2.putText(frame, f"{k}: {v}",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 0), 2)
            y += 35

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("✅ Conteo finalizado")

# -----------------------------------------------------------
# TABLA FINAL
# -----------------------------------------------------------
st.subheader("📊 Conteo final por tipo de vehículo")

df = pd.DataFrame({
    "Tipo de Vehículo": list(st.session_state.count_by_class.keys()),
    "Cantidad": list(st.session_state.count_by_class.values())
})

st.dataframe(df, use_container_width=True)
