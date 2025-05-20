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

# Logo en esquina superior izquierda
col_logo, _ = st.columns([1, 10])
with col_logo:
    st.image("logo.png", width=80)

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

# Funciones internas

def find_dicom_series(directory):
    series = []
    for root, _, files in os.walk(directory):
        try:
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for sid in ids:
                flist = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, sid)
                if flist:
                    series.append((sid, root, flist))
        except:
            pass
    return series


def apply_window_level(image, ww, wc):
    img_f = image.astype(float)
    mn = wc - ww/2.0
    mx = wc + ww/2.0
    win = np.clip(img_f, mn, mx)
    return (win - mn) / (mx - mn) if mx!=mn else np.zeros_like(img_f)

# Extraer ZIP
dirname = None
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as z:
        z.extractall(temp_dir)
    dirname = temp_dir
    st.sidebar.success("Archivos extraídos correctamente.")

# Buscar y cargar serie
img = None
original = None
if dirname:
    with st.spinner('Buscando series DICOM...'):
        series = find_dicom_series(dirname)
    if series:
        opts = [f"Serie {i+1}: {s[0][:10]}... ({len(s[2])} ficheros)" for i,s in enumerate(series)]
        sel = st.sidebar.selectbox("Seleccionar serie DICOM:", opts)
        idx = opts.index(sel)
        _, _, files = series[idx]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        vol = reader.Execute()
        img = sitk.GetArrayViewFromImage(vol)
        original = img.copy()
    else:
        st.sidebar.error("No se encontraron DICOM válidos en el ZIP cargado.")

# Mostrar tres vistas (axial, coronal, sagital) sin panel de selección
def show_views(volume):
    n_ax, n_cor, n_sag = volume.shape
    # Índices centrales
    ia, ic, is_ = n_ax//2, n_cor//2, n_sag//2
    views = [
        ('Axial', volume[ia, :, :]),
        ('Coronal', volume[:, ic, :]),
        ('Sagital', volume[:, :, is_])
    ]
    cols = st.columns(3)
    for col, (name, mat) in zip(cols, views):
        with col:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots()
            ax.axis('off')
            norm = apply_window_level(mat, float(mat.max()-mat.min()), float((mat.max()+mat.min())/2))
            ax.imshow(norm, cmap='gray', origin='lower')
            st.pyplot(fig)

if img is not None:
    show_views(img)
    # Opción de vista 3D still in sidebar
    if st.sidebar.checkbox('Mostrar visualización 3D', value=True):
        target=(64,64,64)
        resized = resize(original, target, anti_aliasing=True)
        x,y,z = np.mgrid[0:target[0],0:target[1],0:target[2]]
        fig3d = go.Figure(data=go.Volume(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=resized.flatten(),
            opacity=0.1, surface_count=15, colorscale='Gray'
        ))
        fig3d.update_layout(margin=dict(l=0,r=0,b=0,t=0))
        st.subheader('Vista 3D')
        st.plotly_chart(fig3d, use_container_width=True)

# Pie de página
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:14px;">
    Brachyanalysis - Visualizador de imágenes DICOM
</div>
""", unsafe_allow_html=True)
