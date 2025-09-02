# -*- coding: utf-8 -*-
# app_cronotrigo_predweem_web.py
# CRONOTrigo + PREDWEEM integrado por WEB (sin OCR/imagenes)
# - Modo Iframe (embed)
# - Modo Extraer tabla (requests + BeautifulSoup)
# - Modo HTML subido (lee tu .html guardado)
# Tabla de riesgos identificada como id="table-riesgos" en el HTML oficial.

import io, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ======= NEW: web components + scraping =======
import streamlit.components.v1 as components
import requests
from bs4 import BeautifulSoup

# ================== UI ==================
st.set_page_config(page_title="CRONOTrigo + PREDWEEM (Web)", layout="wide")
st.markdown("""
<style>
#MainMenu{visibility:hidden} footer{visibility:hidden}
header [data-testid="stToolbar"]{visibility:hidden}
.viewerBadge_container__1QSob,.stAppDeployButton{display:none}
</style>
""", unsafe_allow_html=True)
st.title("CRONOTrigo + PREDWEEM · Integración Web (sin OCR)")

# ================== Utils ==================
def _norm_col(df, aliases):
    for a in aliases:
        if a in df.columns: return a
    return None

def _num(s, pct=False):
    if s is None: return float("nan")
    m = re.search(r"(-?\d+(?:[.,]\d+)?)", str(s).replace(",", "."))
    if not m: return float("nan")
    v = float(m.group(1))
    return v/100.0 if pct and v > 1 else v

def rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2],16); g = int(hex_color[2:4],16); b = int(hex_color[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ================== CRONOTRIGO (WEB) ==================
CRONOTRIGO_URL = "https://cronotrigo.agro.uba.ar/index.php/cronos/AR"

def _scrape_cronotrigo_table(html_text: str) -> pd.DataFrame | None:
    """Devuelve la tabla de riesgos como DataFrame (id='table-riesgos') o None si no se encuentra."""
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table", {"id": "table-riesgos"})  # presente en el HTML oficial:contentReference[oaicite:1]{index=1}
    if not table:
        return None

    # headers
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    # rows
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        rows.append([td.get_text(strip=True) for td in tds])

    if not rows:
        return None

    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    if not headers:
        headers = [f"Columna {i+1}" for i in range(max_len)]
    else:
        headers = headers[:max_len] + [f"Columna {i+1}" for i in range(len(headers), max_len)]

    df = pd.DataFrame(rows, columns=headers)
    return df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_cronotrigo_html() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(CRONOTRIGO_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

# ================== PREDWEEM ==================
THR_BAJO_MEDIO = 0.020
THR_MEDIO_ALTO = 0.079
EMEAC_MIN_DEN, EMEAC_ADJ_DEN, EMEAC_MAX_DEN = 1.8, 2.1, 2.5

GITHUB_WEIGHTS = "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main"
FNAME_IW, FNAME_BIW, FNAME_LW, FNAME_BOUT = "IW.npy", "bias_IW.npy", "LW.npy", "bias_out.npy"

@st.cache_data(ttl=900, show_spinner=False)
def load_public_csv():
    urls = [
        "https://PREDWEEM.github.io/ANN/meteo_daily.csv",
        "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/meteo_daily.csv"
    ]
    for url in urls:
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            req = {"Fecha","Julian_days","TMAX","TMIN","Prec"}
            if req.issubset(df.columns):
                return df.sort_values("Fecha").reset_index(drop=True)
        except Exception:
            continue
    raise RuntimeError("No se pudo cargar el CSV público de meteo.")

def _sanitize_meteo(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Julian_days","TMAX","TMIN","Prec"]
    out = df.copy()
    for c in cols: out[c] = pd.to_numeric(out[c], errors="coerce")
    out[cols] = out[cols].interpolate(limit_direction="both")
    out["Julian_days"] = out["Julian_days"].clip(1, 366)
    out["Prec"] = out["Prec"].clip(lower=0)
    m = out["TMAX"] < out["TMIN"]
    if m.any():
        out.loc[m, ["TMAX","TMIN"]] = out.loc[m, ["TMIN","TMAX"]].values
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_npy(url):
    from urllib.request import urlopen, Request
    with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=20) as resp:
        raw = resp.read()
    return np.load(io.BytesIO(raw), allow_pickle=False)

def _cargar_pesos():
    IW = _fetch_npy(f"{GITHUB_WEIGHTS}/{FNAME_IW}")
    b1 = _fetch_npy(f"{GITHUB_WEIGHTS}/{FNAME_BIW}")
    LW = _fetch_npy(f"{GITHUB_WEIGHTS}/{FNAME_LW}")
    bout = _fetch_npy(f"{GITHUB_WEIGHTS}/{FNAME_BOUT}").item()
    assert IW.shape[0]==4 and LW.shape[0]==1 and LW.shape[1]==IW.shape[1]
    return IW, b1, LW, float(bout)

class PracticalANNModel:
    def __init__(self, IW, b1, LW, b2):
        self.IW, self.b1, self.LW, self.b2 = IW, b1, LW, float(b2)
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)
    def _tansig(self, x): return np.tanh(x)
    def _norm(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1
    def _denorm_out(self, y, ymin=-1, ymax=1):
        return (y - ymin) / (ymax - ymin)
    def predict(self, X_real):
        Xn = self._norm(X_real)
        z1 = Xn @ self.IW + self.b1
        a1 = self._tansig(z1)
        z2 = (a1 @ self.LW.T).ravel() + self.b2
        y  = self._tansig(z2)
        y  = self._denorm_out(y)  # 0..1
        ac = np.cumsum(y) / 8.05
        diff = np.diff(ac, prepend=0)
        niveles = np.where(diff <= THR_BAJO_MEDIO, "Bajo",
                   np.where(diff <= THR_MEDIO_ALTO, "Medio", "Alto"))
        return pd.DataFrame({"EMERREL(0-1)": diff, "Nivel": niveles})

def run_predweem_simple(df_meteo: pd.DataFrame):
    df = _sanitize_meteo(df_meteo).copy()
    if "Fecha" not in df.columns or not np.issubdtype(df["Fecha"].dtype, np.datetime64):
        year = pd.Timestamp.now().year
        df["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
    IW, b1, LW, b2 = _cargar_pesos()
    model = PracticalANNModel(IW, b1, LW, b2)
    X = df[["Julian_days","TMIN","TMAX","Prec"]].to_numpy(float)
    out = model.predict(X)
    out["Fecha"] = pd.to_datetime(df["Fecha"])
    out["Julian_days"] = df["Julian_days"].values
    out["EMERREL acumulado"] = out["EMERREL(0-1)"].cumsum().clip(upper=1.0)
    out["MA5"] = out["EMERREL(0-1)"].rolling(5, min_periods=1).mean()
    return out

def run_predweem_from_file(pred_file):
    ext = Path(pred_file.name).suffix.lower()
    df = pd.read_excel(pred_file) if ext in (".xls",".xlsx") else pd.read_csv(pred_file)
    col_f = _norm_col(df, ["Fecha","date","Day","dia"])
    if not col_f: raise ValueError("Archivo PREDWEEM debe incluir una columna de Fecha.")
    df["Fecha"] = pd.to_datetime(df[col_f], dayfirst=True, errors="coerce")
    col_e = _norm_col(df, ["EMERREL","EMERAC","EmergenciaRel","emerrel","emerac"])
    if not col_e: raise ValueError("No se encontró EMERREL/EMERAC.")
    ser = pd.to_numeric(df[col_e], errors="coerce").fillna(0.0)
    if ser.max() > 1.0001: ser = ser / ser.max()
    out = pd.DataFrame({"Fecha": pd.to_datetime(df["Fecha"]), "EMERREL(0-1)": ser}).sort_values("Fecha")
    out["EMERREL acumulado"] = out["EMERREL(0-1)"].cumsum().clip(upper=1.0)
    out["MA5"] = out["EMERREL(0-1)"].rolling(5, min_periods=1).mean()
    if "Julian_days" in df.columns:
        out["Julian_days"] = pd.to_numeric(df["Julian_days"], errors="coerce")
    return out.reset_index(drop=True)

# ================== MeteoBahía (opcional) ==================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_meteobahia_xml(url: str) -> str:
    from urllib.request import urlopen, Request
    with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def parse_meteobahia_xml(xml_text: str) -> pd.DataFrame:
    try:
        from lxml import etree as ET
        root = ET.fromstring(xml_text.encode("utf-8"))
    except Exception:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text.encode("utf-8"))
    rows = []
    candidates = root.findall(".//dia") + root.findall(".//day") + root.findall(".//item")
    if not candidates: candidates = root.findall(".//*")
    for node in candidates:
        def _grab(taglist):
            for k in taglist:
                el = node.find(k)
                if el is not None and el.text: return el.text.strip()
                if node.get(k): return node.get(k)
            return None
        date_txt = _grab(["fecha","date","day","dia","f"])
        tmax = _grab(["tmax","TMAX","max","tx"])
        tmin = _grab(["tmin","TMIN","min","tn"])
        rain = _grab(["rain","prec","lluvia","pp","pr"])
        if not date_txt: continue
        dt = pd.to_datetime(date_txt, dayfirst=True, errors="coerce")
        rows.append({"Fecha": dt, "TMAX": _num(tmax), "TMIN": _num(tmin), "Prec": max(_num(rain),0.0) if rain else 0.0})
    df = pd.DataFrame(rows).dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    if df.empty: raise ValueError("XML sin días parseables.")
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df = _sanitize_meteo(df[["Fecha","Julian_days","TMAX","TMIN","Prec"]])
    return df

# ================== Sidebar ==================
with st.sidebar:
    st.header("CRONOTRIGO (Web)")
    modo_crono = st.radio(
        "Elegí el modo de integración",
        ["Iframe (recomendado)", "Extraer tabla (vivo)", "Usar HTML subido"],
        index=0
    )

    cronot_html_file = None
    if modo_crono == "Usar HTML subido":
        cronot_html_file = st.file_uploader(
            "Subí el HTML exportado de Cronotrigo",
            type=["html","htm"],
            key="cronot_html"
        )

    st.markdown("---")
    st.header("PREDWEEM")
    modo_pred = st.radio("Origen de datos", ["Subir archivo (EMERREL/EMERAC)", "CSV público", "API MeteoBahía (XML)"], index=0)
    pred_file = None; meteo_url = None
    if modo_pred == "Subir archivo (EMERREL/EMERAC)":
        pred_file = st.file_uploader("CSV/XLSX diario", type=["csv","xlsx"], key="pred_up")
    elif modo_pred == "API MeteoBahía (XML)":
        meteo_url = st.text_input("URL XML", value="https://meteobahia.com.ar/scripts/forecast/for-bd.xml")

# ================== CRONOTRIGO: Visualización / Datos ==================
st.subheader("CRONOTRIGO – Resultados FAUBA")

cronot_df = None  # DataFrame con tabla de riesgos si se pudo extraer
if modo_crono == "Iframe (recomendado)":
    components.iframe(CRONOTRIGO_URL, height=900, scrolling=True)
    st.caption("Si el sitio bloquea iframes, usá ‘Extraer tabla (vivo)’ o ‘Usar HTML subido’.")
    st.link_button("Abrir Cronotrigo en pestaña nueva", CRONOTRIGO_URL)

elif modo_crono == "Extraer tabla (vivo)":
    with st.spinner("Consultando Cronotrigo…"):
        try:
            html_text = fetch_cronotrigo_html()
            cronot_df = _scrape_cronotrigo_table(html_text)
            if cronot_df is not None:
                st.success("Tabla de riesgos extraída correctamente.")
                st.dataframe(cronot_df, use_container_width=True)
            else:
                st.warning("No se encontró la tabla en la página. Probá ‘Usar HTML subido’.")
        except Exception as e:
            st.error(f"No pude leer Cronotrigo: {e}")

else:  # "Usar HTML subido"
    if cronot_html_file is not None:
        try:
            html_text = cronot_html_file.read().decode("utf-8", errors="ignore")
            cronot_df = _scrape_cronotrigo_table(html_text)
            if cronot_df is not None:
                st.success("Tabla de riesgos leída desde el HTML subido.")
                st.dataframe(cronot_df, use_container_width=True)
            else:
                st.warning("No encontré la tabla en este HTML. Asegurate de subir la página correcta.")
        except Exception as e:
            st.error(f"No pude procesar el HTML subido: {e}")
    else:
        st.info("Subí el archivo HTML para continuar.")

# ================== PREDWEEM ==================
st.subheader("Serie PREDWEEM")
pred_vis = None
try:
    if modo_pred == "Subir archivo (EMERREL/EMERAC)":
        if pred_file:
            pred_vis = run_predweem_from_file(pred_file); st.success(f"Serie cargada: {len(pred_vis)} días.")
        else:
            st.info("Subí un archivo con Fecha y EMERREL/EMERAC.")
    elif modo_pred == "CSV público":
        df_meteo = load_public_csv(); pred_vis = run_predweem_simple(df_meteo)
        st.success(f"PREDWEEM corrido con meteo pública: {len(pred_vis)} días.")
    else:
        if meteo_url and meteo_url.strip():
            xml_text = fetch_meteobahia_xml(meteo_url.strip())
            df_meteo = parse_meteobahia_xml(xml_text)
            pred_vis = run_predweem_simple(df_meteo)
            st.success(f"PREDWEEM corrido con API MeteoBahía: {len(pred_vis)} días.")
        else:
            st.info("Ingresá la URL del XML de MeteoBahía.")
except Exception as e:
    st.error(f"No se pudo generar la serie de PREDWEEM: {e}")

# ================== Gráficos ==================
def colores_por_nivel(serie, pal=("Bajo","#2ca02c"), pb=("Medio","#ff7f0e"), pa=("Alto","#d62728")):
    mp = {pal[0]: pal[1], pb[0]: pb[1], pa[0]: pa[1]}
    return serie.map(mp).fillna("#808080").to_numpy()

fig_er = fig_ac = None
if pred_vis is not None and len(pred_vis):
    pred_plot = pred_vis.copy()
    if "Nivel" not in pred_plot.columns:
        th1, th2 = THR_BAJO_MEDIO, THR_MEDIO_ALTO
        pred_plot["Nivel"] = np.where(pred_plot["EMERREL(0-1)"] <= th1, "Bajo",
                               np.where(pred_plot["EMERREL(0-1)"] <= th2, "Medio", "Alto"))

    st.subheader("EMERREL diario (MA5 + sombreado)")
    colors = colores_por_nivel(pred_plot["Nivel"])
    fig_er = go.Figure()
    fig_er.add_trace(go.Scatter(x=pred_plot["Fecha"], y=pred_plot["MA5"], mode="lines",
                                line=dict(width=0), fill="tozeroy", fillcolor=rgba("#4169e1",0.15),
                                showlegend=False, hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=pred_plot["Fecha"], y=pred_plot["MA5"], mode="lines",
                                line=dict(width=2), name="MA5",
                                hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"))
    fig_er.add_bar(x=pred_plot["Fecha"], y=pred_plot["EMERREL(0-1)"],
                   marker=dict(color=colors.tolist()),
                   customdata=pred_plot["Nivel"].map({"Bajo":"🟢 Bajo","Medio":"🟠 Medio","Alto":"🔴 Alto"}),
                   hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
                   name="EMERREL (0-1)")
    fig_er.add_hline(y=THR_BAJO_MEDIO, line_dash="dot", opacity=0.6, annotation_text=f"Bajo ≤ {THR_BAJO_MEDIO:.3f}")
    fig_er.add_hline(y=THR_MEDIO_ALTO, line_dash="dot", opacity=0.6, annotation_text=f"Medio ≤ {THR_MEDIO_ALTO:.3f}")
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)", hovermode="x unified",
                         height=520, legend_title="Referencias")
    st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    st.subheader("EMEAC (%)")
    emeac = pd.DataFrame({
        "Fecha": pred_plot["Fecha"],
        "EMEAC_min": (pred_plot["EMERREL acumulado"]/EMEAC_MIN_DEN*100).clip(0,100),
        "EMEAC_adj": (pred_plot["EMERREL acumulado"]/EMEAC_ADJ_DEN*100).clip(0,100),
        "EMEAC_max": (pred_plot["EMERREL acumulado"]/EMEAC_MAX_DEN*100).clip(0,100)
    })
    fig_ac = go.Figure()
    fig_ac.add_trace(go.Scatter(x=emeac["Fecha"], y=emeac["EMEAC_max"], mode="lines", line=dict(width=0), name="Máximo"))
    fig_ac.add_trace(go.Scatter(x=emeac["Fecha"], y=emeac["EMEAC_min"], mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo"))
    fig_ac.add_trace(go.Scatter(x=emeac["Fecha"], y=emeac["EMEAC_adj"], mode="lines", line=dict(width=2.5), name=f"Ajustable (/{EMEAC_ADJ_DEN:.2f})"))
    for nivel in [25,50,75,90]:
        try: fig_ac.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
        except Exception: pass
    fig_ac.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", hovermode="x unified",
                         height=480, legend_title="Referencias")
    st.plotly_chart(fig_ac, use_container_width=True, theme="streamlit")

# ================== Descargas ==================
st.subheader("Descargas")
cols = st.columns(3)

# Descarga tabla CRONOTRIGO (si se obtuvo)
if cronot_df is not None:
    buf_ct = io.StringIO()
    cronot_df.to_csv(buf_ct, index=False)
    cols[0].download_button("⬇ Tabla CRONOTRIGO (CSV)", data=buf_ct.getvalue(),
                            file_name="cronotrigo_tabla_riesgos.csv", mime="text/csv")

# Serie PREDWEEM (si existe)
if pred_vis is not None:
    buf3 = io.StringIO(); pred_vis.to_csv(buf3, index=False)
    cols[1].download_button("⬇ Serie PREDWEEM (CSV)", data=buf3.getvalue(),
                            file_name="predweem_serie.csv", mime="text/csv")

# Paquete ZIP (gráficos + datos disponibles)
def fig_to_html_bytes(fig):
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

zip_ready = (cronot_df is not None) or (pred_vis is not None)
if zip_ready:
    with io.BytesIO() as mem:
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if cronot_df is not None:
                _b = io.StringIO(); cronot_df.to_csv(_b, index=False)
                zf.writestr("cronotrigo_tabla_riesgos.csv", _b.getvalue())
            if pred_vis is not None:
                _b = io.StringIO(); pred_vis.to_csv(_b, index=False)
                zf.writestr("predweem_serie.csv", _b.getvalue())
            if 'fig_er' in locals() and fig_er is not None:
                zf.writestr("grafico_emerrel.html", fig_to_html_bytes(fig_er))
            if 'fig_ac' in locals() and fig_ac is not None:
                zf.writestr("grafico_emeac.html", fig_to_html_bytes(fig_ac))
        mem.seek(0)
        cols[2].download_button("⬇ Descargar TODO (ZIP)", data=mem.read(),
                                file_name="cronotrigo_predweem_paquete.zip", mime="application/zip")
