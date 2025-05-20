import os
import sys
import io
import zipfile
import tempfile
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from skimage.transform import resize
import plotly.graph_objects as go
import pydicom

# Configuración de la página
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

# Cargar dinámicamente REFERENCE.py
app_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(app_dir, '..'))
candidate_paths = [
    os.path.join(app_dir, 'REFERENCE.py'),
    os.path.join(project_root, 'REFERENCE.py'),
    '/mnt/data/REFERENCE.py'
]
reference_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if not reference_path:
    st.error('No se encontró REFERENCE.py en rutas conocidas: ' + ', '.join(candidate_paths))
    st.stop()
spec = importlib.util.spec_from_file_location('REFERENCE', reference_path)
REFERENCE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REFERENCE)

# Alias a funciones clave
extract_zip = REFERENCE.extract_zip
find_dicom_series = REFERENCE.find_dicom_series
load_dicom_series = REFERENCE.load_dicom_series
load_rtstruct = REFERENCE.load_rtstruct
compute_needle_trajectories = REFERENCE.compute_needle_trajectories
draw_slice = REFERENCE.draw_slice
draw_3d_visualization = REFERENCE.draw_3d_visualization

# Sidebar: carga de ZIP
st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('Sube un archivo ZIP con tus archivos DICOM', type='zip')

if uploaded_file:
    # Extraer contenido del ZIP
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    st.sidebar.success('Archivos extraídos correctamente.')

    # Buscar series DICOM
    with st.spinner('Buscando series DICOM...'):
        series = find_dicom_series(temp_dir)
    if not series:
        st.sidebar.error('No se encontraron series DICOM válidas en el ZIP.')
    else:
        sid, root_dir, file_list = series[0]
        volume, volume_info = load_dicom_series(file_list)

        # Cargar estructuras RTSTRUCT
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
        structures = load_rtstruct(rtstruct_path) if rtstruct_path else {}
        if not rtstruct_path:
            st.sidebar.warning('No se encontró RTSTRUCT; estructuras deshabilitadas.')

        # Parámetros CTV y agujas
        st.sidebar.markdown('<p class="sidebar-title">Configuración de CTV y Agujas</p>', unsafe_allow_html=True)
        num_needles = st.sidebar.slider('Número de agujas', 1, 12, 6)
        cyl_d = st.sidebar.number_input('Diámetro del cilindro CT (mm)', 1.0, 100.0, 20.0, 0.1)
        cyl_l = st.sidebar.number_input('Longitud del cilindro CT (mm)', 1.0, 200.0, 50.0, 0.1)
        offsets = [
            st.sidebar.number_input('Offset CTV X (mm)', value=0.0, step=0.1),
            st.sidebar.number_input('Offset CTV Y (mm)', value=0.0, step=0.1),
            st.sidebar.number_input('Offset CTV Z (mm)', value=0.0, step=0.1)
        ]

        # Calcular centroid CTV
        ctv_centroid = None
        if 'CTV' in structures:
            pts = np.vstack([c['points'] for c in structures['CTV']['contours']])
            ctv_centroid = pts.mean(axis=0) + np.array(offsets)
        else:
            st.sidebar.warning("Estructura 'CTV' no encontrada.")

        # Trayectorias de agujas
        entries, trajectories = compute_needle_trajectories(
            num_needles, cyl_d, cyl_l, structures, volume_info, offsets
        )

        # Opciones de visualización
        st.sidebar.markdown('<p class="sidebar-title">Opciones de visualización</p>', unsafe_allow_html=True)
        show_struct = st.sidebar.checkbox('Mostrar estructuras', True)
        show_needles = st.sidebar.checkbox('Mostrar trayectorias de agujas', True)
        show_cyl2d = st.sidebar.checkbox('Mostrar cilindro 2D', True)
        lw = st.sidebar.slider('Grosor de líneas', 1, 8, 2)

        # Selección de corte y renderizado
        corte = st.sidebar.radio('Selecciona el tipo de corte', ['Axial','Coronal','Sagital'])
        idx = {
            'Axial': st.sidebar.slider('Índice Axial', 0, volume.shape[0]-1, volume.shape[0]//2),
            'Coronal': st.sidebar.slider('Índice Coronal', 0, volume.shape[1]-1, volume.shape[1]//2),
            'Sagital': st.sidebar.slider('Índice Sagital', 0, volume.shape[2]-1, volume.shape[2]//2)
        }[corte]
        planes = ['axial','coronal','sagittal']
        cols = st.columns(3)
        for col, pl in zip(cols, planes):
            with col:
                st.markdown(f'*{pl.capitalize()}*')
                fig = draw_slice(
                    volume, idx if pl.lower()==corte.lower() else None,
                    pl, structures, volume_info,
                    needle_trajectories=trajectories if show_needles else [],
                    cylinder_diameter=cyl_d, cylinder_length=cyl_l,
                    ctv_centroid=ctv_centroid, show_structures=show_struct,
                    show_cylinder_2d=show_cyl2d, linewidth=lw
                )
                st.pyplot(fig)

        # Vista 3D
        st.subheader('Vista 3D')
        fig3d = draw_3d_visualization(
            structures, trajectories if show_needles else [],
            volume_info, cyl_d, cyl_l, ctv_centroid
        )
        if fig3d:
            st.plotly_chart(fig3d, use_container_width=True)

# Footer
st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:14px;">
    Brachyanalysis - Visualizador de imágenes DICOM con estructuras y planificación
</div>
""", unsafe_allow_html=True)
