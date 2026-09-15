import streamlit as st
import numpy as np
import pyvista as pv
import plotly.graph_objects as go

st.set_page_config(page_title="Generador de Hélices Paramétricas 3D", layout="wide")

st.title("⚓ Generador de Hélices Paramétricas 3D")
st.markdown("Modifica los parámetros geométricos en la barra lateral para reconstruir la hélice en tiempo real.")

pv.OFF_SCREEN = True

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
    diametro_cono_ent = st.sidebar.slider("Diámetro Cono Entrada", min_value=0.1*D, max_value=0.5*D, value=0.25*D)
    diametro_cono_sal = st.sidebar.slider("Diámetro Cono Salida", min_value=0.05*D, max_value=0.4*D, value=0.15*D)

st.sidebar.subheader("📐 Relación Paso / Diámetro (P/D)")
PD_global = st.sidebar.slider("Paso / Diámetro (P/D)", min_value=0.5, max_value=1.8, value=1.0, step=0.05)

# ==========================================
# CÁLCULO GEOMÉTRICO
# ==========================================
def obtener_parametros(tipo, fskw_num):
    if tipo == 1: # Kaplan (Ka)
        angrake = 0.0
        r_R_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        K = [1.168, 1.322, 1.508, 1.677, 1.831, 1.970, 2.084, 2.167, 2.218]
        CSkew = [0.050, 0.028, 0.013, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0]
        CSkew = [c * fskw_num for c in CSkew]
        Ctmax = [0.0450, 0.0400, 0.0352, 0.0300, 0.0245, 0.0190, 0.0138, 0.0092, 0.0061]
    else: # Wageningen B (Termina en punta redondeada en r/R=1.0)
        angrake = np.pi / 12.0
        r_R_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.0]
        K = [1.415, 1.662, 1.882, 2.050, 2.152, 2.187, 2.144, 1.970, 1.200, 0.050]
        CSkew = [0.122, 0.117, 0.113, 0.101, 0.086, 0.061, 0.024, -0.037, -0.100, -0.149]
        CSkew = [c * fskw_num for c in CSkew]
        Ctmax = [0.0408, 0.0366, 0.0324, 0.0282, 0.0240, 0.0198, 0.0156, 0.0114, 0.0050, 0.0010]
    
    return angrake, r_R_list, K, CSkew, Ctmax

def generar_malla_pala(D_val, Z_val, FaF_val, tipo, fskw_val, pd_val):
    angrake, r_R_list, K, CSkew, Ctmax = obtener_parametros(tipo, fskw_val)
    n_secciones = len(K)
    
    secciones_pts = []
    
    for i in range(n_secciones):
        rsR = r_R_list[i]
        paso = pd_val * D_val
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

def generar_datos_cono_trunco_cerrado(r_entrada, r_salida, largo, n_p=32):
    """Genera superficie lateral y tapas planas (entrada y salida) para cerrar el cono"""
    y_vals = np.array([-largo/2, largo/2])
    r_vals = np.array([r_entrada, r_salida])
    theta = np.linspace(0, 2*np.pi, n_p, endpoint=False)
    
    x = list(np.outer(r_vals, np.cos(theta)).flatten())
    z = list(np.outer(r_vals, np.sin(theta)).flatten())
    y = list(np.repeat(y_vals, n_p))
    
    # Agregar centros para las tapas
    x.extend([0.0, 0.0])
    y.extend([-largo/2, largo/2])
    z.extend([0.0, 0.0])
    
    idx_centro_inf = 2 * n_p
    idx_centro_sup = 2 * n_p + 1
    
    i_list, j_list, k_list = [], [], []
    
    for idx in range(n_p):
        next_idx = (idx + 1) % n_p
        r0_curr = idx
        r0_next = next_idx
        r1_curr = idx + n_p
        r1_next = next_idx + n_p
        
        # Malla lateral
        i_list.extend([r0_curr, r0_next])
        j_list.extend([r1_curr, r1_curr])
        k_list.extend([r0_next, r1_next])
        
        # Tapa Entrada
        i_list.append(idx_centro_inf)
        j_list.append(r0_next)
        k_list.append(r0_curr)
        
        # Tapa Salida
        i_list.append(idx_centro_sup)
        j_list.append(r1_curr)
        k_list.append(r1_next)
        
    return x, y, z, i_list, j_list, k_list

# ==========================================
# CONSTRUCCIÓN DE LA ESCENA EN PLOTLY
# ==========================================
fig = go.Figure()

# 1. Cono Central Cerrado
largo_cono = D * 0.4
r_ent = diametro_cono_ent / 2.0
r_sal = diametro_cono_sal / 2.0

x_c, y_c, z_c, i_c, j_c, k_c = generar_datos_cono_trunco_cerrado(r_ent, r_sal, largo_cono)

fig.add_trace(go.Mesh3d(
    x=x_c, y=y_c, z=z_c,
    i=i_c, j=j_c, k=k_c,
    color="gold", opacity=1.0, name="Cono Central", flatshading=True
))

# 2. Palas de la Hélice
pala_pts = generar_malla_pala(D, Z, FaF, tipo_helice, fskw, PD_global)

n_sec, n_pts, _ = pala_pts.shape
grid = pv.StructuredGrid()
grid.points = pala_pts.reshape(-1, 3)
grid.dimensions = [n_pts, n_sec, 1]
grid_poly = grid.extract_surface().triangulate()

angulo_pala = 360.0 / Z
for i in range(Z):
    pala_rotada = grid_poly.rotate_y(i * angulo_pala, inplace=False)
    pts = pala_rotada.points
    faces = pala_rotada.faces.reshape(-1, 4)[:, 1:]
    
    fig.add_trace(go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="deepskyblue", opacity=1.0, name=f"Pala {i+1}", flatshading=True
    ))

fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

# ==========================================
# DESPLIEGUE EN STREAMLIT
# ==========================================
col_view, col_info = st.columns([3, 1])

with col_view:
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("📊 Métricas Estimadas")
    volumen_est = (np.pi * (D/2)**2 * 0.05) * FaF * (Z / 4)
    st.metric(label="Diámetro Exterior", value=f"{D:.2f} m")
    st.metric(label="Paso Hélice (P)", value=f"{(PD_global * D):.2f} m")
    st.metric(label="Relación P/D", value=f"{PD_global:.2f}")
    st.metric(label="Volumen Aprox. Palas", value=f"{volumen_est:.4f} m³")
    st.metric(label="Peso Aprox. (Bronce)", value=f"{volumen_est * 8500:.1f} kg")
