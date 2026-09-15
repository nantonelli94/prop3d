import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from scipy.interpolate import CubicSpline
import pyvista as pv
import tempfile

st.set_page_config(page_title="Generador de Hélices Paramétricas 3D", layout="wide")

st.title("⚓ Generador de Hélices Paramétricas 3D")
st.markdown("Modifica los parámetros geométricos en la barra lateral para reconstruir la hélice en tiempo real.")

# Configurar PyVista para servidores Headless
pv.start_xvfb()

# ==========================================
# BARRA LATERAL: CONTROLES
# ==========================================
st.sidebar.header("⚙️ Parámetros de la Hélice")

tipo_helice = st.sidebar.radio("Tipo de Hélice", [1, 2], format_func=lambda x: "Serie B de Wageningen" if x == 2 else "Serie Ka (Kaplan)")

col1, col2 = st.sidebar.columns(2)
with col1:
    D = st.sidebar.slider("Diámetro D [m]", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    Z = st.sidebar.slider("Número de Palas (Z)", min_value=2, max_value=7, value=4, step=1)
    FaF = st.sidebar.slider("Relación Áreas (Fa/F)", min_value=0.3, max_value=1.1, value=0.55, step=0.05)

with col2:
    fskw = st.sidebar.slider("Factor Skew (Inclinación)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    diametro_cono_ent = st.sidebar.slider("Diámetro Cono Entrada", min_value=0.1*D, max_value=0.4*D, value=0.2*D)
    diametro_cono_sal = st.sidebar.slider("Diámetro Cono Salida", min_value=0.05*D, max_value=0.3*D, value=0.15*D)

st.sidebar.subheader("📐 Perfil Paso / Diámetro (P/D)")
pd02 = st.sidebar.slider("P/D r/R=0.2", 0.5, 1.8, 1.0, 0.05)
pd07 = st.sidebar.slider("P/D r/R=0.7", 0.5, 1.8, 1.0, 0.05)
pd10 = st.sidebar.slider("P/D r/R=1.0", 0.5, 1.8, 1.0, 0.05)

# Interpolación P/D
x_pd = np.array([0.2, 0.7, 1.0])
y_pd = np.array([pd02, pd07, pd10])
poly_pd = np.polyfit(x_pd, y_pd, 2)

def get_pd(r_R):
    return poly_pd[0]*(r_R**2) + poly_pd[1]*r_R + poly_pd[2]

# ==========================================
# CÁLCULO GEOMÉTRICO
# ==========================================
def obtener_parametros(tipo, Z_num, FaF_num, fskw_num):
    if tipo == 1: # Kaplan
        angrake = 0.0
        K = [1.168, 1.322, 1.508, 1.677, 1.831, 1.970, 2.084, 2.167, 2.218, 2.219]
        CSkew = [0.050, 0.028, 0.013, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        CSkew = [c * fskw_num for c in CSkew]
        Ctmax = [0.0450, 0.0400, 0.0352, 0.0300, 0.0245, 0.0190, 0.0138, 0.0092, 0.0061, 0.0050]
    else: # Wageningen B
        angrake = np.pi / 12.0
        K = [1.415, 1.662, 1.882, 2.050, 2.152, 2.187, 2.144, 1.970, 1.582]
        CSkew = [0.122, 0.117, 0.113, 0.101, 0.086, 0.061, 0.024, -0.037, -0.149]
        CSkew = [c * fskw_num for c in CSkew]
        Ctmax = [0.0408, 0.0366, 0.0324, 0.0282, 0.0240, 0.0198, 0.0156, 0.0114, 0.0072]
    
    return angrake, K, CSkew, Ctmax

def generar_malla_pala(D_val, Z_val, FaF_val, tipo, fskw_val):
    angrake, K, CSkew, Ctmax = obtener_parametros(tipo, Z_val, FaF_val, fskw_val)
    n_secciones = len(K)
    
    secciones_pts = []
    
    for i in range(n_secciones):
        rsR = (i + 1.0) / 10.0
        PD = get_pd(rsR)
        paso = PD * D_val
        alfa = np.arctan(paso / (np.pi * rsR * D_val))
        
        c = (D_val * FaF_val * K[i]) / Z_val
        skew = c * CSkew[i]
        tmax = D_val * Ctmax[i]
        rake = np.tan(angrake) * rsR * (D_val / 2.0)
        
        t = np.linspace(0, 1, 30)
        x_perfil = c * (t - 0.5)
        
        yt = 5 * tmax * (0.2969*np.sqrt(t) - 0.1260*t - 0.3516*t**2 + 0.2843*t**3 - 0.1015*t**4)
        
        x_pts = np.concatenate([x_perfil, x_perfil[::-1]])
        y_pts = np.concatenate([yt, -yt[::-1]])
        
        x_pts += skew
        
        x_rot = x_pts * np.cos(alfa) - y_pts * np.sin(alfa)
        y_rot = x_pts * np.sin(alfa) + y_pts * np.cos(alfa)
        
        r = (D_val / 2.0) * rsR
        theta = x_rot / r
        
        X_final = r * np.cos(theta)
        Z_final = r * np.sin(theta)
        Y_final = y_rot - rake
        
        pts_3d = np.vstack((X_final, Y_final, Z_final)).T
        secciones_pts.append(pts_3d)
        
    return np.array(secciones_pts)

# ==========================================
# RENDERIZADO 3D CON PYVISTA (EXPORTADO A HTML)
# ==========================================
plotter = pv.Plotter(notebook=False)
plotter.set_background("#1e1e1e")

# 1. Cono Central
largo_cono = D * 0.4
cono = pv.Cone(center=(0, 0, 0), direction=(0, 1, 0), 
               height=largo_cono, radius=diametro_cono_ent/2.0, resolution=32)
plotter.add_mesh(cono, color="gold", metallic=0.8, smooth_shading=True)

# 2. Palas
pala_pts = generar_malla_pala(D, Z, FaF, tipo_helice, fskw)

n_sec, n_pts, _ = pala_pts.shape
grid = pv.StructuredGrid()
grid.points = pala_pts.reshape(-1, 3)
grid.dimensions = [n_pts, n_sec, 1]

angulo_pala = 360.0 / Z
for i in range(Z):
    pala_rotada = grid.rotate_y(i * angulo_pala, inplace=False)
    plotter.add_mesh(pala_rotada, color="cyan", show_edges=False, metallic=0.5, smooth_shading=True)

# Exportar a HTML interactivo sin hilos/multiprocessing
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
    path_html = tf.name
    plotter.export_html(path_html)

# ==========================================
# MOSTRAR EN STREAMLIT
# ==========================================
col_view, col_info = st.columns([3, 1])

with col_view:
    with open(path_html, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=False)

with col_info:
    st.subheader("📊 Métricas Estimadas")
    volumen_est = (np.pi * (D/2)**2 * 0.05) * FaF * (Z / 4)
    st.metric(label="Diámetro Exterior", value=f"{D:.2f} m")
    st.metric(label="Paso a r/R=0.7", value=f"{(pd07*D):.2f} m")
    st.metric(label="Volumen Aprox. Palas", value=f"{volumen_est:.4f} m³")
    st.metric(label="Peso Aprox. (Bronce)", value=f"{volumen_est * 8500:.1f} kg")
