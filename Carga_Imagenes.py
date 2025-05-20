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
import pydicom

# Importar funciones desde REFERENCE.py
from REFERENCE import (
    extract_zip,
    find_dicom_series,
    load_dicom_series,
    load_rtstruct,
    compute_needle_trajectories,
    draw_slice,
    draw_3d_visualization
)

st.set_page_config(layout="wide", page_title="Brachyanalysis")

# Estilos personalizados
st.markdown("""
<style>
    .giant-title { color: #28aec5; text-align: center; font-size: 72px; margin: 30px 0; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { color: #c0d711; font-size: 24px; margin-bottom: 15px; font-weight: bold; }
    .sidebar-title { color: #28aec5; font-size: 20px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
    .stButton>button { background-color: #28aec5; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
    .stButton>button:hover { background-color: #1c94aa; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

if uploaded_file:
    # Extraer ZIP
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    st.sidebar.success("Archivos extraídos correctamente.")

    # Buscar series DICOM
    with st.spinner('Buscando series DICOM...'):
        series = find_dicom_series(temp_dir)
    if not series:
        st.sidebar.error("No se encontraron series DICOM válidas en el ZIP.")
    else:
        sid, root_dir, file_list = series[0]

        # Cargar volumen y metadatos
        volume, volume_info = load_dicom_series(file_list)

        # Cargar RTSTRUCT y estructuras
        rtstruct_path = None
        for rd, dirs, files in os.walk(temp_dir):
            for fname in files:
                if fname.lower().endswith('.dcm'):
                    ds = pydicom.dcmread(os.path.join(rd, fname), stop_before_pixels=True)
                    if getattr(ds, 'Modality', '') == 'RTSTRUCT':
                        rtstruct_path = os.path.join(rd, fname)
                        break
            if rtstruct_path:
                break

        if rtstruct_path:
            structures = load_rtstruct(rtstruct_path)
        else:
            structures = {}
            st.sidebar.warning("No se encontró un RTSTRUCT para extraer estructuras.")

        # Configuración de CTV y agujas
        st.sidebar.markdown('<p class="sidebar-title">Configuración de CTV y Agujas</p>', unsafe_allow_html=True)
        num_needles = st.sidebar.slider("Número de agujas", min_value=1, max_value=12, value=6)
        cylinder_diameter = st.sidebar.number_input("Diámetro del cilindro CT (mm)", min_value=1.0, max_value=100.0, value=20.0, step=0.1)
        cylinder_length = st.sidebar.number_input("Longitud del cilindro CT (mm)", min_value=1.0, max_value=200.0, value=50.0, step=0.1)
        ctv_offset_x = st.sidebar.number_input("Offset CTV X (mm)", value=0.0, step=0.1)
        ctv_offset_y = st.sidebar.number_input("Offset CTV Y (mm)", value=0.0, step=0.1)
        ctv_offset_z = st.sidebar.number_input("Offset CTV Z (mm)", value=0.0, step=0.1)

        # Calcular centroide CTV
        ctv_centroid = None
        if 'CTV' in structures:
            all_pts = np.vstack([c['points'] for c in structures['CTV']['contours']])
            ctv_centroid = all_pts.mean(axis=0) + np.array([ctv_offset_x, ctv_offset_y, ctv_offset_z])
        else:
            st.sidebar.warning("No se encontró la estructura 'CTV' para calcular el cilindro.")

        # Calcular trayectorias de agujas
        needle_entries, needle_trajectories = compute_needle_trajectories(
            num_needles,
            cylinder_diameter,
            cylinder_length,
            structures,
            volume_info,
            [ctv_offset_x, ctv_offset_y, ctv_offset_z]
        )

        st.session_state.needle_entries = needle_entries
        st.session_state.needle_trajectories = needle_trajectories
        st.session_state.ctv_centroid = ctv_centroid

        # Opciones de visualización
        st.sidebar.markdown('<p class="sidebar-title">Opciones de visualización</p>', unsafe_allow_html=True)
        show_structures = st.sidebar.checkbox("Mostrar estructuras", value=True)
        show_needle_trajectories = st.sidebar.checkbox("Mostrar trayectorias de agujas", value=True)
        show_cylinder_2d = st.sidebar.checkbox("Mostrar cilindro 2D", value=True)
        linewidth = st.sidebar.slider("Grosor de líneas", min_value=1, max_value=8, value=2)

        # Selección de corte
        corte = st.sidebar.radio("Selecciona el tipo de corte", ("Axial", "Coronal", "Sagital"))
        if corte == "Axial":
            slice_idx = st.sidebar.slider("Índice Axial", 0, volume.shape[0] - 1, volume.shape[0] // 2)
            plane = 'axial'
        elif corte == "Coronal":
            slice_idx = st.sidebar.slider("Índice Coronal", 0, volume.shape[1] - 1, volume.shape[1] // 2)
            plane = 'coronal'
        else:
            slice_idx = st.sidebar.slider("Índice Sagital", 0, volume.shape[2] - 1, volume.shape[2] // 2)
            plane = 'sagittal'

        # Mostrar cortes con estructuras, trayectorias y cilindro
        cols = st.columns(3)
        for col, pl, idx in zip(cols, ['axial', 'coronal', 'sagittal'], [slice_idx if plane=='axial' else volume.shape[0]//2,
                                                                            slice_idx if plane=='coronal' else volume.shape[1]//2,
                                                                            slice_idx if plane=='sagittal' else volume.shape[2]//2]):
            with col:
                st.markdown(f"*{pl.capitalize()}*")
                fig = draw_slice(
                    volume,
                    idx,
                    pl,
                    structures,
                    volume_info,
                    needle_trajectories=needle_trajectories if show_needle_trajectories else [],
                    cylinder_diameter=cylinder_diameter,
                    cylinder_length=cylinder_length,
                    ctv_centroid=ctv_centroid,
                    show_structures=show_structures,
                    show_cylinder_2d=show_cylinder_2d,
                    linewidth=linewidth
                )
                st.pyplot(fig)

        # Visualización 3D
        st.subheader("Vista 3D")
        fig3d = draw_3d_visualization(
            structures,
            needle_trajectories if show_needle_trajectories else [],
            volume_info,
            cylinder_diameter,
            cylinder_length,
            ctv_centroid
        )
        if fig3d:
            st.plotly_chart(fig3d, use_container_width=True)

# Título y pie de página
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:14px;">
    Brachyanalysis - Visualizador de imágenes DICOM con estructuras y planificación
</div>
""", unsafe_allow_html=True)
