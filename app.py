import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Analizador de Emblemas",
    page_icon="⚫",
    layout="wide"
)

st.title("🔎 Analizador de Porcentaje de Área Negra en Emblemas")
st.write(
    "Sube una imagen, dibuja un polígono alrededor del emblema para seleccionarlo, "
    "y la aplicación calculará el porcentaje del área negra dentro de esa selección."
)

# --- Funciones de Procesamiento de Imagen ---

def calcular_porcentaje_negro(imagen, puntos):
    """
    Calcula el porcentaje de área negra dentro de un polígono en una imagen.

    Args:
        imagen (np.array): La imagen original en formato OpenCV (BGR).
        puntos (list): Una lista de tuplas (x, y) con las coordenadas del polígono.

    Returns:
        tuple: Una tupla conteniendo el porcentaje (float), la imagen resultado (np.array),
               y el área total del polígono (int).
    """
    if not puntos:
        return 0.0, imagen, 0

    # Crear una máscara a partir de los puntos seleccionados por el usuario
    mascara_poligono = np.zeros(imagen.shape[:2], dtype=np.uint8)
    puntos_np = np.array(puntos, dtype=np.int32)
    cv2.fillPoly(mascara_poligono, [puntos_np], 255)

    # Convertir a escala de grises y aplicar umbral para detectar el color negro
    # Un umbral bajo (ej. 50) considera negros y grises muy oscuros.
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, mascara_negro = cv2.threshold(gris, 50, 255, cv2.THRESH_BINARY_INV)

    # Intersecar la máscara del polígono con la máscara de píxeles negros
    mascara_final_negro = cv2.bitwise_and(mascara_negro, mascara_poligono)

    # Calcular áreas
    area_total_poligono = cv2.countNonZero(mascara_poligono)
    area_negra = cv2.countNonZero(mascara_final_negro)

    if area_total_poligono == 0:
        return 0.0, imagen, 0

    porcentaje = (area_negra / area_total_poligono) * 100

    # Crear una imagen de resultado visualmente clara
    # Fondo gris, con el área negra detectada en negro
    fondo_gris = np.full(imagen.shape, (200, 200, 200), dtype=np.uint8)
    resultado_color = cv2.cvtColor(mascara_final_negro, cv2.COLOR_GRAY2BGR)
    resultado_final = np.where(resultado_color == 255, (0, 0, 0), fondo_gris)
    
    # Dibuja el contorno seleccionado en la imagen resultado
    cv2.polylines(resultado_final, [puntos_np], isClosed=True, color=(0, 255, 0), thickness=2)

    return porcentaje, resultado_final, area_total_poligono


# --- Interfaz de la Aplicación ---

# Configuración de columnas para la interfaz
col1, col2 = st.columns(spec=[0.6, 0.4], gap="large")

with col1:
    st.header("1. Carga y Selecciona el Área")
    uploaded_file = st.file_uploader(
        "Elige una imagen de emblema:", type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_file is not None:
        # Leer la imagen subida y convertirla a un formato compatible con OpenCV
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_original = np.array(pil_image)
        # Convertir de RGB (PIL) a BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

        # Determinar el tamaño del lienzo de dibujo
        CANVAS_WIDTH = 600
        escala = CANVAS_WIDTH / img_bgr.shape[1]
        canvas_height = int(img_bgr.shape[0] * escala)

        st.write("Dibuja un polígono sobre la imagen. Haz doble clic en el primer punto para cerrar la forma.")
        
        # Crear el lienzo para dibujar sobre la imagen
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=pil_image,
            update_streamlit=True,
            width=CANVAS_WIDTH,
            height=canvas_height,
            drawing_mode="polygon",
            key="canvas",
        )

# La lógica de procesamiento se ejecuta fuera de las columnas para usar los datos del lienzo
if uploaded_file is not None and canvas_result.json_data is not None and canvas_result.json_data["objects"]:
    # Extraer los puntos del polígono dibujado por el usuario
    puntos_canvas = canvas_result.json_data["objects"][0]["path"]
    # Los puntos son una lista de listas, ej: [['M', x1, y1], ['L', x2, y2], ...]. Extraemos (x,y)
    puntos_reducidos = [tuple(p[1:]) for p in puntos_canvas]
    
    # Escalar los puntos del lienzo al tamaño de la imagen original
    puntos_originales = [(int(x / escala), int(y / escala)) for x, y in puntos_reducidos]

    # Calcular el porcentaje y obtener la imagen de resultado
    porcentaje, img_resultado, area_total = calcular_porcentaje_negro(img_bgr, puntos_originales)
    
    with col2:
        st.header("2. Resultados del Análisis")
        if area_total > 0:
            # Mostrar la métrica del porcentaje
            st.metric(
                label="Porcentaje de Negro en el Área Seleccionada",
                value=f"{porcentaje:.2f} %"
            )

            # Mostrar la imagen de resultado (convertir de BGR a RGB para Streamlit)
            img_resultado_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
            st.image(img_resultado_rgb, caption="Resultado: Área negra detectada (contorno verde)")

            # Crear botón de descarga
            # Primero, codificar la imagen a un formato de archivo en memoria
            _, buffer = cv2.imencode('.png', img_resultado)
            
            # Obtener nombre de archivo original para usarlo en la descarga
            nombre_archivo_original = uploaded_file.name
            nombre_salida = f"resultado_{int(porcentaje)}pct_{nombre_archivo_original}"

            st.download_button(
                label="📥 Descargar Imagen de Resultado",
                data=buffer.tobytes(),
                file_name=nombre_salida,
                mime="image/png"
            )
        else:
            st.warning("El área seleccionada es demasiado pequeña o no se ha dibujado un polígono. Inténtalo de nuevo.")
else:
    with col2:
        st.info("Esperando a que subas una imagen y dibujes un polígono...")
