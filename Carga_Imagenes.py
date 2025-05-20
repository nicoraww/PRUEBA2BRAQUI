import os
import io
import zipfile
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from skimage.transform import resize
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Brachyanalysis")

# Estilos
st.markdown("""
<style>
    .giant-title { color: #28aec5; text-align: center; font-size: 72px; margin: 30px 0; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { color: #c0d711; font-size: 24px; margin-bottom: 15px; font-weight: bold; }
    .stButton>button { background-color: #28aec5; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
    .stButton>button:hover { background-color: #1c94aa; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

def find_dicom_series(directory):
    series_found = []
    for root, dirs, files in os.walk(directory):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for sid in series_ids:
                file_list = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, sid)
                if file_list:
                    series_found.append((sid, root, file_list))
        except Exception:
            continue
    return series_found

def apply_window_level(image, window_width, window_center):
    img_float = image.astype(float)
    min_v = window_center - window_width / 2.0
    max_v = window_center + window_width / 2.0
    windowed = np.clip(img_float, min_v, max_v)
    if max_v != min_v:
        return (windowed - min_v) / (max_v - min_v)
    return np.zeros_like(img_float)

dirname = None
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    dirname = temp_dir
    st.sidebar.success("Archivos extraídos correctamente.")

dicom_series = None
img = None
original_image = None
if dirname:
    with st.spinner('Buscando series DICOM...'):
        dicom_series = find_dicom_series(dirname)
    if dicom_series:
        options = [f"Serie {i + 1}: {series[0][:10]}... ({len(series[2])} archivos)" for i, series in enumerate(dicom_series)]
        selection = st.sidebar.selectbox("Seleccionar serie DICOM:", options)
        selected_idx = options.index(selection)
        sid, dirpath, files = dicom_series[selected_idx]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        data = reader.Execute()
        img = sitk.GetArrayViewFromImage(data)
        original_image = img
    else:
        st.sidebar.error("No se encontraron DICOM válidos en el ZIP cargado.")

if img is not None:
    n_ax, n_cor, n_sag = img.shape
    min_val, max_val = float(img.min()), float(img.max())
    default_ww = max_val - min_val
    default_wc = min_val + default_ww / 2
    ww, wc = default_ww, default_wc

    corte = st.sidebar.radio("Selecciona el tipo de corte", ("Axial", "Coronal", "Sagital"))

    if corte == "Axial":
        corte_idx = st.sidebar.slider("Selecciona el índice axial", 0, n_ax - 1, n_ax // 2)
        axial_img = img[corte_idx, :, :]
        coronal_img = img[:, n_cor // 2, :]
        sagital_img = img[:, :, n_sag // 2]
    elif corte == "Coronal":
        corte_idx = st.sidebar.slider("Selecciona el índice coronal", 0, n_cor - 1, n_cor // 2)
        coronal_img = img[:, corte_idx, :]
        axial_img = img[n_ax // 2, :, :]
        sagital_img = img[:, :, n_sag // 2]
    elif corte == "Sagital":
        corte_idx = st.sidebar.slider("Selecciona el índice sagital", 0, n_sag - 1, n_sag // 2)
        sagital_img = img[:, :, corte_idx]
        axial_img = img[n_ax // 2, :, :]
        coronal_img = img[:, n_cor // 2, :]

    # Mostrar imágenes 2D en una fila
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("*Axial*")
        fig1, ax1 = plt.subplots()
        ax1.axis('off')
        ax1.imshow(apply_window_level(axial_img, ww, wc), cmap='gray', origin='lower')
        st.pyplot(fig1)

    with col2:
        st.markdown("*Coronal*")
        fig2, ax2 = plt.subplots()
        ax2.axis('off')
        ax2.imshow(apply_window_level(coronal_img, ww, wc), cmap='gray', origin='lower')
        st.pyplot(fig2)

    with col3:
        st.markdown("*Sagital*")
        fig3, ax3 = plt.subplots()
        ax3.axis('off')
        ax3.imshow(apply_window_level(sagital_img, ww, wc), cmap='gray', origin='lower')
        st.pyplot(fig3)

    # Imagen 3D
    target_shape = (64, 64, 64)
    img_resized = resize(original_image, target_shape, anti_aliasing=True)
    x, y, z = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
    fig3d = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=img_resized.flatten(),
        opacity=0.1,
        surface_count=15,
        colorscale="Gray",
    ))
    fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    st.subheader("Vista 3D")
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:14px;">
    Brachyanalysis - Visualizador de imágenes DICOM
</div>
""", unsafe_allow_html=True)
