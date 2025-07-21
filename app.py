import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Contorno de Emblemas", layout="wide")

st.title("Dibuja el contorno del emblema")
st.write("Usa la herramienta de 'Free draw' para delimitar el contorno")

uploaded_file = st.file_uploader("Sube una imagen del emblema", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.subheader("Dibuja sobre la imagen")
    canvas_resultado = st_canvas(
        fill_color="rgba(255, 255, 0, 0.3)",  # Amarillo semitransparente
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=image,
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_resultado.image_data is not None:
        mask = np.array(canvas_resultado.image_data)

        # Convertimos a escala de grises y luego a binario
        gray = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contorno_img = image_np.copy()
        cv2.drawContours(contorno_img, contours, -1, (0, 255, 0), 2)

        st.subheader("Contorno detectado")
        st.image(contorno_img, caption="Contorno del emblema", use_column_width=True)

        # Puedes guardar si lo deseas
        if st.button("Guardar resultado"):
            resultado = Image.fromarray(contorno_img)
            resultado.save("resultado.png")
            st.success("Imagen guardada como resultado.png")
