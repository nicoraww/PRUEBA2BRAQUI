import os
import io
import zipfile
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import SimpleITK as sitk

# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="Brachyanalysis")
st.markdown("""
<style>
    .giant-title { color: #28aec5; text-align: center; font-size: 48px; margin-bottom: 10px; font-weight: bold; }
    .sub-header { color: #28aec5; font-size: 20px; margin-bottom: 5px; font-weight: bold; }
    .sidebar-title { color: #28aec5; font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .plot-container { padding: 5px; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)

# Barra lateral de configuración
st.sidebar.markdown('<p class="sidebar-title">Configuración DICOM</p>', unsafe_allow_html=True)
zip_file = st.sidebar.file_uploader("Selecciona ZIP con tus archivos DICOM", type="zip")

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_first_series(uploaded):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded.read()), 'r') as zf:
        zf.extractall(tmpdir)
    for root, _, _ in os.walk(tmpdir):
        ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
        if ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, ids[0])
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            img3d = reader.Execute()
            return sitk.GetArrayViewFromImage(img3d)
    return None

# Cargar volumen DICOM
img = None
if zip_file:
    with st.spinner('Cargando serie DICOM...'):
        img = load_first_series(zip_file)
        # Convertir a numpy array para asegurar compatibilidad
        if img is not None:
            img = np.asarray(img)
    if img is None:
        st.sidebar.error("No se encontró ninguna serie DICOM válida en el ZIP.")

# Función de ventana/nivel/nivel
def window_image(slice2d, ww, wl):
    arr = slice2d.astype(float)
    mn = wl - ww/2.0
    mx = wl + ww/2.0
    clipped = np.clip(arr, mn, mx)
    return (clipped - mn)/(mx - mn) if mx != mn else np.zeros_like(arr)

# Renderizado de un slice
def render_slice(slice2d, ww, wl):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(window_image(slice2d, ww, wl), cmap='gray', origin='lower')
    ax.axis('off')
    return fig

# Mostrar cuadrícula de vistas 2D
if img is not None:
    # Asegurar que el volumen sea 3D (incluir dimensión de slices)
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    nz, ny, nx = img.shape
    nz, ny, nx = img.shape
    # Sliders de cortes
    st.sidebar.subheader('Cortes')
    z_ix = st.sidebar.slider('Axial', 0, nz-1, nz//2)
    y_ix = st.sidebar.slider('Coronal', 0, ny-1, ny//2)
    x_ix = st.sidebar.slider('Sagital', 0, nx-1, nx//2)

    # Controles de ventana/nivel
    st.sidebar.subheader('Ventana y Nivel (WW/WL)')
    min_val, max_val = float(img.min()), float(img.max())
    default_ww = max_val - min_val if max_val>min_val else 1.0
    default_wl = (max_val + min_val) / 2
    ww = st.sidebar.number_input('WW', min_value=1.0, value=default_ww)
    wl = st.sidebar.number_input('WL', value=default_wl)

    # Cuadrícula de tres vistas
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader('Axial')
        st.pyplot(render_slice(img[z_ix,:,:], ww, wl))
    with c2:
        st.subheader('Coronal')
        st.pyplot(render_slice(img[:,y_ix,:], ww, wl))
    with c3:
        st.subheader('Sagital')
        st.pyplot(render_slice(img[:,:,x_ix], ww, wl))

    # Pie de página
    st.markdown('---')
    st.markdown('<div style="text-align:center;color:#28aec5;font-size:14px;">Brachyanalysis - 2D Quadrants Viewer</div>', unsafe_allow_html=True)
