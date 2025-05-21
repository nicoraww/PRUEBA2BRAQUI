import os
import io
import zipfile
import tempfile
import random

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import SimpleITK as sitk
from skimage.transform import resize
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide", page_title="BrachyCervix")

# Logo en esquina superior izquierda
col1, col2 = st.columns([5, 15])
with col1:
    st.image("Banner.png", width=500)

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
    return (win - mn) / (mx - mn) if mx != mn else np.zeros_like(img_f)

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
        opts = [f"Serie {i+1}: {s[0][:10]}... ({len(s[2])} ficheros)" for i, s in enumerate(series)]
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

# Si hay imagen cargada
if img is not None:
    n_ax, n_cor, n_sag = img.shape
    mn, mx = float(img.min()), float(img.max())
    default_ww, default_wc = mx - mn, mn + (mx - mn) / 2

    # Seleccionar cortes
    sync = st.sidebar.checkbox('Sincronizar cortes', value=True)
    if sync:
        corte = st.sidebar.radio('Corte (sincronizado)', ('Axial', 'Coronal', 'Sagital'))
        lims = {'Axial': n_ax-1, 'Coronal': n_cor-1, 'Sagital': n_sag-1}
        mids = {'Axial': n_ax//2, 'Coronal': n_cor//2, 'Sagital': n_sag//2}
        idx_slider = st.sidebar.slider('Corte (sincronizado)', 0, lims[corte], mids[corte])
        slice_idx = st.sidebar.number_input('Corte (sincronizado)', 0, lims[corte], idx_slider)
    else:
        corte = st.sidebar.radio('Selecciona el tipo de corte', ('Axial', 'Coronal', 'Sagital'))
        if corte == 'Axial':
            slice_idx = st.sidebar.slider('Índice Axial', 0, n_ax-1, n_ax//2)
        elif corte == 'Coronal':
            slice_idx = st.sidebar.slider('Índice Coronal', 0, n_cor-1, n_cor//2)
        else:
            slice_idx = st.sidebar.slider('Índice Sagital', 0, n_sag-1, n_sag//2)

    # Opciones adicionales
    show_3d = st.sidebar.checkbox('Mostrar visualización 3D', value=True)
    invert = st.sidebar.checkbox('Invertir colores (Negativo)', value=False)
    window_type = st.sidebar.selectbox('Tipo de ventana', ('Default', 'Abdomen', 'Hueso', 'Pulmón'))
    if window_type == 'Default':
        ww, wc = default_ww, default_wc
    elif window_type == 'Abdomen':
        ww, wc = 400, 40
    elif window_type == 'Hueso':
        ww, wc = 2000, 500
    else:
        ww, wc = 1500, -600

    # Preparar cortes 2D
    axial = img[slice_idx, :, :] if corte == 'Axial' else img[n_ax//2, :, :]
    coronal = img[:, slice_idx, :] if corte == 'Coronal' else img[:, n_cor//2, :]
    sagital = img[:, :, slice_idx] if corte == 'Sagital' else img[:, :, n_sag//2]
    cortes = [('Axial', axial), ('Coronal', coronal), ('Sagital', sagital)]

    cols = st.columns(3)
    for col, (name, mat) in zip(cols, cortes):
        with col:
            st.markdown(f"{name}")
            fig, ax = plt.subplots()
            ax.axis('off')
            norm = apply_window_level(mat, ww, wc)
            if invert:
                norm = 1 - norm
            ax.imshow(norm, cmap='gray', origin='lower')
            st.pyplot(fig)

    # Visualización 3D con agujas agregables
    if show_3d:
        resized = resize(original, (64, 64, 64), anti_aliasing=True)

        if 'needles' not in st.session_state:
            st.session_state['needles'] = []

        with st.expander("Agregar aguja 3D"):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                x1 = st.number_input("X1", 0.0, 64.0, 32.0)
                y1 = st.number_input("Y1", 0.0, 64.0, 32.0)
                z1 = st.number_input("Z1", 0.0, 64.0, 32.0)
            with col_a2:
                x2 = st.number_input("X2", 0.0, 64.0, 32.0)
                y2 = st.number_input("Y2", 0.0, 64.0, 32.0)
                z2 = st.number_input("Z2", 0.0, 64.0, 32.0)
            if st.button("Agregar Aguja", key="add_manual"):
                color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                st.session_state['needles'].append({
                    'points': ((x1, y1, z1), (x2, y2, z2)),
                    'color': color
                })
            if st.button("Generar Aguja Aleatoria", key="add_random"):
                # Punto A aleatorio en [7,35], Punto B aleatorio en [30,45]
                xa = random.uniform(7, 35)
                ya = random.uniform(7, 35)
                za = random.uniform(7, 35)
                xb = random.uniform(30, 45)
                yb = random.uniform(30, 45)
                zb = random.uniform(30, 45)
                color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                st.session_state['needles'].append({
                    'points': ((xa, ya, za), (xb, yb, zb)),
                    'color': color
                })

        # Crear figura 3D
        xg, yg, zg = np.mgrid[0:64, 0:64, 0:64]
        fig3d = go.Figure(data=[
            go.Volume(
                x=xg.flatten(), y=yg.flatten(), z=zg.flatten(),
                value=resized.flatten(), opacity=0.1,
                surface_count=15, colorscale='Gray'
            )
        ])

        # Dibujar agujas (puntos y líneas)
        for needle in st.session_state['needles']:
            (x1, y1, z1), (x2, y2, z2) = needle['points']
            col = needle['color']
            fig3d.add_trace(
                go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2],
                    mode='markers+lines',
                    marker=dict(size=5, color=col),
                    line=dict(width=3, color=col)
                )
            )

        # Mostrar vista 3D
        fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.subheader('Vista 3D')
        st.plotly_chart(fig3d, use_container_width=True)

        # Registro de agujas como tabla editable
        st.markdown('### Registro de agujas')
        # Preparar DataFrame con columnas X1,Y1,Z1,X2,Y2,Z2,Color,Eliminar
        rows = []
        for i, needle in enumerate(st.session_state['needles'], start=1):
            (x1, y1, z1), (x2, y2, z2) = needle['points']
            rows.append({
                'ID': i,
                'X1': round(x1, 1),
                'Y1': round(y1, 1),
                'Z1': round(z1, 1),
                'X2': round(x2, 1),
                'Y2': round(y2, 1),
                'Z2': round(z2, 1),
                'Color': needle['color'],
                'Eliminar': False
            })
        df = pd.DataFrame(rows)
        edited_df = st.experimental_data_editor(df, num_rows='dynamic', use_container_width=True)
        # Procesar cambios
        new_needles = []
        for _, row in edited_df.iterrows():
            if not row['Eliminar']:
                pts = ((row['X1'], row['Y1'], row['Z1']), (row['X2'], row['Y2'], row['Z2']))
                new_needles.append({'points': pts, 'color': row['Color']})
        st.session_state['needles'] = new_needles

# Pie de página
st.markdown('<p class="giant-title">BrachyCervix</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:50px;">
    BrachyCervix - Semiautomátización y visor para procesos de braquiterapia enfocados en el Cervix
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    Proyecto asignatura medialab 3
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    Universidad EAFIT 
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    Clínica Las Américas AUNA 
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - Nicolás Ramirez 
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - Alejandra Montiel
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - Maria Camila Diaz
