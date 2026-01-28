# -*- coding: utf-8 -*-

# ===================== IMPORTS =====================

import os
import time
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from sqlalchemy import create_engine, Column, Integer, String, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ===================== CONFIG =====================

VIDEO_FOLDER = "videos"
RECORDINGS_FOLDER = "recordings"
LOGO_PATH = "output-onlinepngtools.png"
PROJECT_TITLE = "Sistema de Detección y Conteo de Vehículos - Tiempo Real y Videos"

CLASS_ID_TO_CATEGORY = {2: "Liviano", 5: "Bus", 7: "Camion"}
CONFIDENCE_THRESHOLD = 0.35
DB_FLUSH_INTERVAL = 5

SOFT_COLORS = [
    "#F2C88A",
    "#F4DE9A", 
    "#156F8E",
    "#15997F",
    "#9DD49E"
]

# ===================== DATABASE =====================

# Variables globales para la base de datos
engine = None
Session = None
traffic_table = None

def init_database():
    """Inicializa la base de datos de forma segura"""
    global engine, Session, traffic_table
    
    try:
        engine = create_engine("sqlite:///traffic.db", echo=False, pool_pre_ping=True)
        metadata = MetaData()
        
        traffic_table = Table(
            "traffic", metadata,
            Column("id", Integer, primary_key=True),
            Column("timestamp", DateTime),
            Column("category", String(50)),
            Column("count", Integer)
        )
        
        metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return True
    except Exception as e:
        st.error(f"Error al inicializar la base de datos: {str(e)}")
        return False

def save_counts_to_db(counts_dict):
    """Guarda los conteos en la base de datos de forma segura"""
    global Session, traffic_table
    
    # Verificar que la base de datos esté inicializada
    if Session is None or traffic_table is None:
        st.warning("Base de datos no inicializada")
        return
    
    session = Session()
    ts = datetime.now()
    try:
        for cat, cnt in counts_dict.items():
            if cnt > 0:
                session.execute(
                    traffic_table.insert().values(
                        timestamp=ts,
                        category=cat,
                        count=cnt
                    )
                )
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Error al guardar en base de datos: {str(e)}")
    finally:
        session.close()

def get_database_data():
    """Obtiene datos de la base de datos de forma segura"""
    global engine
    
    if engine is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql("SELECT * FROM traffic", engine)
        return df
    except Exception as e:
        st.warning(f"No se pudieron cargar los datos históricos: {str(e)}")
        return pd.DataFrame()

# ===================== TRACKER =====================

class SimpleCentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def update(self, detections):
        new_ids = []
        for (x1, y1, x2, y2, cat) in detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            matched = False

            for oid in list(self.objects.keys()):
                ox, oy = self.objects[oid]["centroid"]
                if np.linalg.norm(np.array((cx, cy)) - np.array((ox, oy))) <= self.max_distance:
                    self.objects[oid]["centroid"] = (cx, cy)
                    matched = True
                    break

            if not matched:
                self.objects[self.next_object_id] = {
                    "centroid": (cx, cy),
                    "category": cat
                }
                new_ids.append((self.next_object_id, cat))
                self.next_object_id += 1

        return new_ids

# ===================== FUNCIONES AUXILIARES =====================

def get_available_cameras():
    """Detecta las cámaras disponibles en el sistema"""
    available_cameras = []
    
    # Probar diferentes backends de OpenCV
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends:
        for i in range(5):  # Probar índices 0-4
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Verificar que el frame no esté vacío
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            available_cameras.append((i, backend))
                            st.success(f"✅ Cámara {i} detectada con backend {backend}")
                cap.release()
            except Exception as e:
                continue
        
        if available_cameras:  # Si encontramos cámaras, no probar más backends
            break
    
    # Remover duplicados manteniendo el orden
    unique_cameras = []
    seen_indices = set()
    for cam_idx, backend in available_cameras:
        if cam_idx not in seen_indices:
            unique_cameras.append((cam_idx, backend))
            seen_indices.add(cam_idx)
    
    return unique_cameras

def test_camera_connection(camera_index, backend=cv2.CAP_ANY):
    """Prueba la conexión con una cámara específica"""
    try:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            # Configurar propiedades básicas
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Leer varios frames para asegurar estabilidad
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return True
                time.sleep(0.1)
            
            cap.release()
        return False
    except Exception as e:
        st.error(f"Error al probar cámara: {str(e)}")
        return False

def get_available_videos():
    """Obtiene la lista de videos disponibles"""
    videos = []
    
    # Videos en carpeta videos
    if os.path.exists(VIDEO_FOLDER):
        for f in os.listdir(VIDEO_FOLDER):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                videos.append(("video", os.path.join(VIDEO_FOLDER, f), f))
    
    # Videos grabados
    if os.path.exists(RECORDINGS_FOLDER):
        for f in os.listdir(RECORDINGS_FOLDER):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                videos.append(("recording", os.path.join(RECORDINGS_FOLDER, f), f))
    
    return videos

def process_detection(frame, model, conf, tracker):
    """Procesa la detección en un frame"""
    results = model(frame)[0]
    detections = []
    
    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        confv = float(b.conf[0])
        cls = int(b.cls[0])
        
        if confv >= conf and cls in CLASS_ID_TO_CATEGORY:
            cat = CLASS_ID_TO_CATEGORY[cls]
            detections.append((x1, y1, x2, y2, cat))
            
            # Colores por categoría
            colors = {"Liviano": (21, 111, 142), "Bus": (255, 165, 0), "Camion": (255, 0, 0)}
            color = colors.get(cat, (255, 255, 255))
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta
            label = f"{cat}: {confv:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Actualizar tracker
    new_ids = tracker.update(detections)
    
    return frame, new_ids

# ===================== UI =====================

st.set_page_config(page_title="Sistema de Monitoreo Vehicular Completo", layout="wide")

# Inicializar base de datos al inicio
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = init_database()

# ===================== ESTILO =====================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3, h4 {
    color: #F4DE9A;
    text-align: center;
}
div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 14px;
}
.stButton>button {
    background: linear-gradient(90deg, #156F8E, #15997F);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}
.camera-status {
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
.camera-connected {
    background-color: rgba(0, 255, 0, 0.2);
    border: 1px solid #00ff00;
}
.camera-disconnected {
    background-color: rgba(255, 0, 0, 0.2);
    border: 1px solid #ff0000;
}
.source-tab {
    background-color: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.warning-box {
    background-color: rgba(255, 165, 0, 0.2);
    border: 1px solid #ffa500;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================

c1, c2, c3 = st.columns([1,4,1])

with c1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

with c2:
    st.markdown(f"# {PROJECT_TITLE}")

# Mostrar estado de la base de datos
if not st.session_state.db_initialized:
    st.error("⚠️ Error en la base de datos. Algunas funciones pueden no estar disponibles.")

st.markdown("---")

# ===================== LAYOUT =====================

left, center, right = st.columns([1,2,1])

# ===================== LEFT =====================

with left:
    st.subheader("📊 Conteo Total")
    liv_metric = st.metric("🚗 Livianos", 0)
    bus_metric = st.metric("🚌 Buses", 0)
    cam_metric = st.metric("🚛 Camiones", 0)

# ===================== RIGHT =====================

with right:
    st.subheader("🎛️ Controles")
    
    # Selección de fuente
    st.write("📹 **Seleccionar Fuente:**")
    source_type = st.radio(
        "Tipo de fuente",
        ["📷 Cámara Web", "🎬 Video Guardado"],
        horizontal=True
    )
    
    # ===== CONFIGURACIÓN CÁMARA WEB =====
    if source_type == "📷 Cámara Web":
        st.markdown('<div class="source-tab">', unsafe_allow_html=True)
        st.write("🔍 **Configuración de Cámara Web**")
        
        # Detección de cámaras con información detallada
        with st.spinner("Detectando cámaras..."):
            available_cameras = get_available_cameras()
        
        if not available_cameras:
            st.error("❌ No se detectaron cámaras USB")
            st.markdown("""
            <div class="warning-box">
            💡 <strong>Soluciones:</strong><br>
            • Conecta tu cámara web USB<br>
            • Verifica que los drivers estén instalados<br>
            • Cierra otras aplicaciones que usen la cámara<br>
            • Prueba diferentes puertos USB<br>
            • Reinicia la aplicación
            </div>
            """, unsafe_allow_html=True)
            camera_source = None
            camera_backend = None
        else:
            st.success(f"✅ {len(available_cameras)} cámara(s) detectada(s)")
            
            # Selección de cámara
            camera_options = {}
            for cam_idx, backend in available_cameras:
                backend_name = {
                    cv2.CAP_DSHOW: "DirectShow",
                    cv2.CAP_MSMF: "Media Foundation", 
                    cv2.CAP_V4L2: "V4L2",
                    cv2.CAP_ANY: "Auto"
                }.get(backend, f"Backend {backend}")
                
                if cam_idx == 0:
                    camera_options[f"Cámara {cam_idx} (Principal) - {backend_name}"] = (cam_idx, backend)
                else:
                    camera_options[f"Cámara {cam_idx} (USB) - {backend_name}"] = (cam_idx, backend)
            
            selected_camera_name = st.selectbox("📹 Seleccionar cámara", list(camera_options.keys()))
            camera_source, camera_backend = camera_options[selected_camera_name]
            
            # Prueba de conexión
            if st.button("🔧 Probar cámara"):
                with st.spinner("Probando conexión..."):
                    if test_camera_connection(camera_source, camera_backend):
                        st.markdown('<div class="camera-status camera-connected">✅ Cámara conectada correctamente</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="camera-status camera-disconnected">❌ Error de conexión con la cámara</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== CONFIGURACIÓN VIDEO =====
    else:
        st.markdown('<div class="source-tab">', unsafe_allow_html=True)
        st.write("🎬 **Configuración de Video**")
        
        # Crear carpetas si no existen
        if not os.path.exists(VIDEO_FOLDER):
            os.makedirs(VIDEO_FOLDER)
        if not os.path.exists(RECORDINGS_FOLDER):
            os.makedirs(RECORDINGS_FOLDER)
        
        # Obtener videos disponibles
        available_videos = get_available_videos()
        
        if not available_videos:
            st.error("❌ No hay videos disponibles")
            st.info(f"💡 Coloca videos en las carpetas:\n- `{VIDEO_FOLDER}/` para videos originales\n- `{RECORDINGS_FOLDER}/` para grabaciones")
            video_source = None
        else:
            st.success(f"✅ {len(available_videos)} video(s) disponible(s)")
            
            # Organizar videos por tipo
            video_options = {}
            for vid_type, vid_path, vid_name in available_videos:
                if vid_type == "video":
                    video_options[f"📁 {vid_name}"] = vid_path
                else:
                    video_options[f"🎥 {vid_name}"] = vid_path
            
            selected_video_name = st.selectbox("🎬 Seleccionar video", list(video_options.keys()))
            video_source = video_options[selected_video_name]
            
            # Información del video
            if st.button("ℹ️ Info del video"):
                cap = cv2.VideoCapture(video_source)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    st.info(f"""
                    📊 **Información del Video:**
                    - Resolución: {width}x{height}
                    - FPS: {fps:.1f}
                    - Frames: {frame_count}
                    - Duración: {duration:.1f}s
                    """)
                    cap.release()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== CONFIGURACIÓN COMÚN =====
    st.write("⚙️ **Parámetros de Detección:**")
    conf = st.slider("🎯 Confianza mínima", 0.1, 1.0, CONFIDENCE_THRESHOLD)
    max_dist = st.slider("📏 Distancia tracker (px)", 20, 200, 60)
    
    # Configuración de resolución (solo para cámara)
    if source_type == "📷 Cámara Web":
        st.write("📐 **Resolución:**")
        resolution_options = {
            "640x480": (640, 480),
            "1280x720 (HD)": (1280, 720),
            "1920x1080 (Full HD)": (1920, 1080)
        }
        selected_resolution = st.selectbox("Resolución", list(resolution_options.keys()))
        width, height = resolution_options[selected_resolution]
    
    # Configuración de grabación
    st.write("💾 **Grabación:**")
    save_video = st.checkbox("Guardar video procesado")
    
    # Botones de control
    start = st.button("▶️ Iniciar Detección")
    stop = st.button("⏹️ Detener")

# ===================== CENTER =====================

with center:
    st.subheader("📹 Monitoreo en Tiempo Real")
    video_frame = st.empty()
    status_text = st.empty()

# ===================== VARIABLES DE ESTADO =====================

if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False

if 'total_counts' not in st.session_state:
    st.session_state.total_counts = {"Liviano": 0, "Bus": 0, "Camion": 0}

# ===================== PROCESS =====================

if start and not st.session_state.detection_active:
    # Verificar que hay una fuente seleccionada
    source_available = False
    if source_type == "📷 Cámara Web" and camera_source is not None:
        source_available = True
        input_source = camera_source
        input_backend = camera_backend
        is_camera = True
    elif source_type == "🎬 Video Guardado" and video_source is not None:
        source_available = True
        input_source = video_source
        input_backend = cv2.CAP_ANY
        is_camera = False
    
    if not source_available:
        st.error("❌ No hay fuente disponible. Selecciona una cámara o video válido.")
        st.stop()
    
    st.session_state.detection_active = True
    
    # Inicializar modelo YOLO
    status_text.text("🔄 Cargando modelo YOLO...")
    try:
        model = YOLO("yolov8n.pt")
        model.overrides["classes"] = [2, 5, 7]
        status_text.text("✅ Modelo YOLO cargado")
    except Exception as e:
        st.error(f"❌ Error al cargar YOLO: {str(e)}")
        st.session_state.detection_active = False
        st.stop()
    
    # Inicializar captura
    status_text.text(f"📹 Conectando con {'cámara' if is_camera else 'video'}...")
    cap = cv2.VideoCapture(input_source, input_backend)
    
    if not cap.isOpened():
        st.error(f"❌ No se pudo abrir la {'cámara' if is_camera else 'video'}")
        st.session_state.detection_active = False
        st.stop()
    
    # Configurar resolución solo para cámara
    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar grabación
    video_writer = None
    if save_video:
        if not os.path.exists(RECORDINGS_FOLDER):
            os.makedirs(RECORDINGS_FOLDER)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = "camera" if is_camera else os.path.basename(input_source).split('.')[0]
        output_path = f"{RECORDINGS_FOLDER}/detection_{source_name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (actual_width, actual_height))
    
    # Inicializar tracker y contadores
    tracker = SimpleCentroidTracker(max_distance=max_dist)
    total = st.session_state.total_counts.copy()
    pending = {"Liviano": 0, "Bus": 0, "Camion": 0}
    last_db = time.time()
    frame_count = 0
    
    status_text.text("🚀 Detección iniciada - Presiona 'Detener' para parar")
    
    # Bucle principal de detección
    while st.session_state.detection_active:
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                st.error("❌ Error al leer de la cámara")
            else:
                st.info("✅ Video terminado")
            break
        
        frame_count += 1
        
        # Procesar detección
        processed_frame, new_ids = process_detection(frame, model, conf, tracker)
        
        # Contar nuevos objetos
        for _, cat in new_ids:
            total[cat] += 1
            pending[cat] += 1
        
        # Agregar información en el frame
        info_text = f"Livianos: {total['Liviano']} | Buses: {total['Bus']} | Camiones: {total['Camion']}"
        cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        source_info = f"{'Camara' if is_camera else 'Video'}: {actual_width}x{actual_height}"
        cv2.putText(processed_frame, source_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not is_camera:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(processed_frame, progress, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar frame
        video_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Actualizar métricas
        liv_metric.metric("🚗 Livianos", total["Liviano"])
        bus_metric.metric("🚌 Buses", total["Bus"])
        cam_metric.metric("🚛 Camiones", total["Camion"])
        
        # Guardar video
        if video_writer:
            video_writer.write(processed_frame)
        
        # Guardar en base de datos (solo si está inicializada)
        if st.session_state.db_initialized and time.time() - last_db > DB_FLUSH_INTERVAL:
            save_counts_to_db(pending)
            pending = {"Liviano": 0, "Bus": 0, "Camion": 0}
            last_db = time.time()
        
        # Control de velocidad
        if is_camera:
            time.sleep(0.03)  # Para cámara en tiempo real
        else:
            time.sleep(0.05)  # Para video un poco más lento
        
        # Verificar stop
        if stop:
            st.session_state.detection_active = False
            break
    
    # Limpieza
    cap.release()
    if video_writer:
        video_writer.release()
        st.success(f"✅ Video guardado: {output_path}")
    
    # Guardar conteos finales
    st.session_state.total_counts = total
    if st.session_state.db_initialized:
        save_counts_to_db(pending)
    
    status_text.text("⏹️ Detección completada")

elif stop:
    st.session_state.detection_active = False
    status_text.text("⏹️ Detección detenida por el usuario")

# ===================== ANALISIS =====================

st.markdown("---")
st.header("📈 Panel de Análisis del Tráfico Vehicular")

# Solo mostrar análisis si la base de datos está disponible
if st.session_state.db_initialized:
    # Obtener datos de forma segura
    df = get_database_data()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["Fecha"] = df["timestamp"].dt.date
        df["Hora"] = df["timestamp"].dt.time
        df["minute"] = df["timestamp"].dt.floor("min")
        df["hour"] = df["timestamp"].dt.hour

        # Filtro por fecha
        st.subheader("📅 Filtro por fecha")
        
        fecha_sel = st.date_input(
            "Seleccione una fecha",
            value=df["Fecha"].iloc[-1]
        )

        df_fecha = df[df["Fecha"] == fecha_sel]

        if not df_fecha.empty:
            resumen = (
                df_fecha.groupby("category")["count"]
                .sum()
                .reindex(["Liviano","Bus","Camion"], fill_value=0)
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("🚗 Livianos", int(resumen["Liviano"]))
            c2.metric("🚌 Buses", int(resumen["Bus"]))
            c3.metric("🚛 Camiones", int(resumen["Camion"]))
        else:
            st.warning("⚠️ No hay datos para la fecha seleccionada")

        # Gráficas
        if len(df) > 1:
            try:
                flow = df.groupby("minute")["count"].sum().reset_index()
                if not flow.empty:
                    st.plotly_chart(px.line(flow, x="minute", y="count", title="📈 Flujo vehicular por minuto"), use_container_width=True)

                bycat = df.groupby("category")["count"].sum().reset_index()
                if not bycat.empty:
                    st.plotly_chart(px.bar(bycat, x="category", y="count", title="🚗 Distribución por tipo de vehículo"), use_container_width=True)

                byhour = df.groupby("hour")["count"].sum().reset_index()
                if not byhour.empty:
                    st.plotly_chart(px.bar(byhour, x="hour", y="count", title="⏰ Demanda vehicular por hora"), use_container_width=True)
            except Exception as e:
                st.warning(f"Error al generar gráficas: {str(e)}")

        st.subheader("📋 Tabla de registros recientes")
        try:
            recent_data = df.tail(50)[["id","Fecha","Hora","category","count"]]
            st.dataframe(recent_data, use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"Error al mostrar tabla: {str(e)}")

    else:
        st.info("⚠️ Aún no hay datos registrados. Inicia la detección para comenzar a recopilar datos.")
else:
    st.warning("⚠️ Base de datos no disponible. El análisis histórico no está disponible.")

# ===================== INFORMACIÓN ADICIONAL =====================

with st.expander("ℹ️ Información del Sistema"):
    st.markdown("""
    ### 🚀 **Características del Sistema:**
    
    **📷 Modo Cámara Web:**
    - Detección en tiempo real desde cámara USB
    - Múltiples backends de OpenCV (DirectShow, Media Foundation)
    - Configuración de resolución personalizable
    - Grabación opcional del video procesado
    
    **🎬 Modo Video:**
    - Procesamiento de videos pregrabados
    - Soporte para múltiples formatos (MP4, AVI, MOV, MKV)
    - Videos desde carpeta `videos/` y grabaciones desde `recordings/`
    
    **🔧 Funcionalidades:**
    - Detección automática de vehículos (Livianos, Buses, Camiones)
    - Tracking para evitar conteos duplicados
    - Base de datos SQLite para almacenamiento
    - Análisis estadístico con gráficas interactivas
    - Filtros por fecha y hora
    
    **📊 Tipos de Vehículos Detectados:**
    - 🚗 **Livianos**: Automóviles, SUVs, camionetas pequeñas
    - 🚌 **Buses**: Autobuses, microbuses, vehículos de transporte público
    - 🚛 **Camiones**: Camiones de carga, vehículos pesados
    
    ### 🔧 **Solución de Problemas:**
    
    **Cámara no detectada:**
    - Verifica que la cámara esté conectada correctamente
    - Cierra otras aplicaciones que puedan usar la cámara
    - Prueba diferentes puertos USB
    - Reinstala los drivers de la cámara
    
    **Error de base de datos:**
    - El sistema continuará funcionando sin análisis histórico
    - Verifica permisos de escritura en la carpeta
    - Elimina el archivo `traffic.db` si está corrupto
    """)

# ===================== DIAGNÓSTICO =====================

with st.expander("🔍 Diagnóstico del Sistema"):
    st.write("**Estado de la Base de Datos:**", "✅ Funcionando" if st.session_state.db_initialized else "❌ Error")
    
    if st.button("🔄 Reinicializar Base de Datos"):
        st.session_state.db_initialized = init_database()
        st.rerun()
    
    st.write("**Cámaras Detectadas:**")
    cameras = get_available_cameras()
    if cameras:
        for idx, (cam_idx, backend) in enumerate(cameras):
            backend_name = {
                cv2.CAP_DSHOW: "DirectShow",
                cv2.CAP_MSMF: "Media Foundation", 
                cv2.CAP_V4L2: "V4L2",
                cv2.CAP_ANY: "Auto"
            }.get(backend, f"Backend {backend}")
            st.write(f"- Cámara {cam_idx}: {backend_name}")
    else:
        st.write("- No se detectaron cámaras")
    
    st.write("**Videos Disponibles:**")
    videos = get_available_videos()
    if videos:
        for vid_type, vid_path, vid_name in videos:
            st.write(f"- {vid_name} ({'Original' if vid_type == 'video' else 'Grabación'})")
    else:
        st.write("- No hay videos disponibles")