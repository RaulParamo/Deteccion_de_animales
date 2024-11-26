# app.py
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Cargar el modelo preentrenado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Configurar la interfaz de Streamlit
st.title("Detección de Animales")
st.write("Usa tu cámara para identificar animales (caballo, vaca, oso, jirafa).")

# Acceder a la cámara
run = st.checkbox('Iniciar/Detener cámara')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Error al acceder a la cámara")
        break
    
    # Convertir la imagen de BGR a RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Realizar detección
    results = model(frame)
    
    # Obtener la imagen con las detecciones
    img_with_detections = results.render()[0]
    
    # Mostrar la imagen en la interfaz de Streamlit
    FRAME_WINDOW.image(img_with_detections)
    
camera.release()
