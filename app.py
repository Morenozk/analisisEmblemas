import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="centered")

st.title("Análisis de área negra en emblemas (Wet Out)")

# Subida de imagen
archivo_imagen = st.file_uploader("Sube una imagen del emblema", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    imagen = Image.open(archivo_imagen).convert("RGB")
    imagen_np = np.array(imagen)

    st.subheader("Dibuja el contorno del emblema")
    st.write("Usa la herramienta de 'Free draw' para delimitar el contorno")

    canvas_resultado = st_canvas(
        fill_color="rgba(255, 255, 0, 0.3)",  # Amarillo semitransparente
        stroke_width=2,
        stroke_color="#FFFF00",
        background_image=imagen,
        update_streamlit=True,
        height=imagen_np.shape[0],
        width=imagen_np.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_resultado.image_data is not None and canvas_resultado.json_data is not None:
        if canvas_resultado.json_data["objects"]:
            mask = np.zeros(imagen_np.shape[:2], dtype=np.uint8)

            # Extraer puntos dibujados y construir la máscara
            for obj in canvas_resultado.json_data["objects"]:
                if obj["type"] == "path":
                    puntos = []
                    for punto in obj["path"]:
                        if punto[0] == "M" or punto[0] == "L":
                            puntos.append((int(punto[1]), int(punto[2])))
                    if len(puntos) > 2:
                        cv2.fillPoly(mask, [np.array(puntos)], 255)

            masked_img = cv2.bitwise_and(imagen_np, imagen_np, mask=mask)
            gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
            _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            black_mask = cv2.bitwise_and(black_mask, mask)

            area_total = cv2.countNonZero(mask)
            area_negra = cv2.countNonZero(black_mask)

            if area_total > 0:
                porcentaje = (area_negra / area_total) * 100
                texto_resultado = f"Porcentaje de área negra: {porcentaje:.2f}%"
                st.success(texto_resultado)

                # Crear imagen de resultado sobre fondo gris
                resultado_color = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
                gris_fondo = np.full_like(resultado_color, 200)
                resultado_final = np.where(resultado_color == 0, gris_fondo, resultado_color)

                # Agregar texto
                cv2.putText(resultado_final, texto_resultado, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                st.image(resultado_final, caption="Resultado final", channels="BGR")
            else:
                st.warning("No se detectó ninguna área válida.")
