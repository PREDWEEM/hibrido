# -*- coding: utf-8 -*-
# app_cronotrigo_predweem_overlap_2025_compare.py
# CRONOTRIGO (web) + PREDWEEM con overlap simplificado y comparación:
# - CSV público + archivo histórico (EMERREL/EMERAC) en simultáneo
# - Muestra SOLO "% EMERREL en PC / Total" (para cada serie disponible)
# - No muestra tabla de CRONOTRIGO
# - Elimina el gráfico EMEAC
# - Horizonte fijo 01/02/2025–01/11/2025

import io, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests

# === Import "seguro" BeautifulSoup (fallback si no está) ===
try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except Exception:
    BeautifulSoup = None
    _BS4_OK = False

# ================== UI ==================
st.set_page_config(page_title="CRONOTRIGO + PREDWEEM · Overlap 2025 (Comparación)", layout="wide")
st.markdown("""
<style>
#MainMenu{visibility:hidden} footer{visibility:hidden}
header [data-testid="stToolbar"]{visibility:hidden}
.viewerBadge_container__1QSob,.stAppDeployButton{display:none}
</style>
""", unsafe_allow_html=True)
st.title("CRONOTRIGO + PREDWEEM · Overlap (horizonte 2025) · Comparación")

# ==== Horizonte fijo 2025 ====
HORIZ_INI = pd.Timestamp("2025-02-01")
HORIZ_FIN = pd.Timestamp("2025-11-01")

def clip_horizon(df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    if df is None or len(df)==0 or "Fecha" not in df.columns:
        return df
    m = (df["Fecha"] >= start) & (df["Fecha"] <= end)
    out = df.loc[m].copy()
    if "Fecha" in out.columns:
        out.sort_values("Fecha", inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out

def clip_pc(pc_i: pd.Timestamp | None, pc_f: pd.Timestamp | None,
            start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if pc_i is None or pc_f is None:
        return (pc_i, pc_f)
    a = max(pc_i, start)
    b = min(pc_f, end)
    if b < a:
        return (None, None)
    return (a, b)

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
DATE_PAT_HTML = re.compile(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?")

def _scrape_cronotrigo_table(html_text: str) -> pd.DataFrame | None:
    """Devuelve la tabla de riesgos (id='table-riesgos') o None si no está / falta bs4."""
    if not _BS4_OK:
        return None
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table", {"id": "table-riesgos"})
    if not table:
        return None
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds: continue
        rows.append([td.get_text(strip=True) for td in tds])
    if not rows:
        return None
    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    if not headers:
        headers = [f"Columna {i+1}" for i in range(max_len)]
    else:
        headers = headers[:max_len] + [f"Columna {i+1}" for i in range(len(headers), max_len)]
    return pd.DataFrame(rows, columns=headers)

def _extract_pc_from_html_text(html_text: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Intenta encontrar 'Período crítico' en el HTML (heurística con 2 fechas cercanas)."""
    text = BeautifulSoup(html_text, "html.parser").get_text(" ", strip=True) if _BS4_OK else html_text
    idx = text.lower().find("período crítico")
    if idx == -1:
        idx = text.lower().find("periodo critico")
    if idx == -1:
        return (None, None)
    start = max(0, idx - 200)
    end = min(len(text), idx + 200)
    snippet = text[start:end]
    dates = DATE_PAT_HTML.findall(snippet)
    if len(dates) < 2:
        dates = DATE_PAT_HTML.findall(text)
        if len(dates) < 2:
            return (None, None)
    year = pd.Timestamp.now().year
    def mkdate(t):
        d, m, y = int(t[0]), int(t[1]), t[2]
        if y:
            y = int(y) if len(y) == 4 else 2000 + int(y)
        else:
            y = year
        try:
            return pd.Timestamp(year=y, month=m, day=d)
        except Exception:
            return pd.NaT
    d1 = mkdate(dates[0]); d2 = mkdate(dates[1])
    if pd.isna(d1) or pd.isna(d2):
        return (None, None)
    if d2 < d1: d1, d2 = d2, d1
    return (d1, d2)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_cronotrigo_html() -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36"}
    resp = requests.get(CRONOTRIGO_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text

# ================== PREDWEEM ==================
THR_BAJO_MEDIO = 0.020
THR_MEDIO_ALTO = 0.079

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
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
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
    def _norm(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1
    def _denorm_out(self, y, ymin=-1, ymax=1):  # tansig^-1 normalizado a 0..1
        return (y - ymin) / (ymax - ymin)
    def predict(self, X_real, IW, b1, LW, b2):
        Xn = self._norm(X_real)
        z1 = Xn @ IW + b1
        a1 = np.tanh(z1)
        z2 = (a1 @ LW.T).ravel() + b2
        y  = np.tanh(z2)
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
    out = model.predict(X, IW, b1, LW, b2)
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
        dt_ = pd.to_datetime(date_txt, dayfirst=True, errors="coerce")
        rows.append({"Fecha": dt_, "TMAX": _num(tmax), "TMIN": _num(tmin), "Prec": max(_num(rain),0.0) if rain else 0.0})
    df = pd.DataFrame(rows).dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    if df.empty: raise ValueError("XML sin días parseables.")
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df = _sanitize_meteo(df[["Fecha","Julian_days","TMAX","TMIN","Prec"]])
    return df

# ================== Sidebar ==================
with st.sidebar:
    st.header("CRONOTRIGO (Web)")
    modo_crono = st.radio(
        "Modo de integración",
        ["Iframe (recomendado)", "Extraer tabla (vivo)", "Usar HTML subido"],
        index=0
    )
    cronot_html_file = None
    if modo_crono == "Usar HTML subido":
        cronot_html_file = st.file_uploader(
            "HTML exportado de CRONOTRIGO",
            type=["html","htm"],
            key="cronot_html"
        )

    st.markdown("---")
    st.header("Período Crítico (PC)")
    st.caption("Si no se detecta desde el HTML, podés ingresarlo manualmente.")
    pc_manual_on = st.checkbox("Ingresar PC manualmente si no se detecta", value=True)
    pc_ini_manual = st.date_input("Inicio PC (manual)", value=None, format="DD/MM/YYYY")
    pc_fin_manual = st.date_input("Fin PC (manual)", value=None, format="DD/MM/YYYY")

    st.markdown("---")
    st.header("PREDWEEM")
    modo_pred = st.radio("Origen de datos base", ["CSV público", "Subir archivo (EMERREL/EMERAC)", "API MeteoBahía (XML)"], index=0)

    # Archivo histórico opcional para comparar (disponible en todos los modos)
    hist_file = st.file_uploader("Archivo histórico opcional (CSV/XLSX) para comparar", type=["csv","xlsx"], key="hist_up")

    pred_file = None; meteo_url = None
    if modo_pred == "Subir archivo (EMERREL/EMERAC)":
        pred_file = st.file_uploader("Archivo base (CSV/XLSX)", type=["csv","xlsx"], key="pred_up")
    elif modo_pred == "API MeteoBahía (XML)":
        meteo_url = st.text_input("URL XML", value="https://meteobahia.com.ar/scripts/forecast/for-bd.xml")

# ================== CRONOTRIGO: Visualización / PC ==================
st.subheader("CRONOTRIGO – Resultados FAUBA")
st.caption("Horizonte aplicado: 01/02/2025 → 01/11/2025")

cronot_df = None
pc_inicio = None
pc_fin = None

if modo_crono == "Iframe (recomendado)":
    components.iframe(CRONOTRIGO_URL, height=900, scrolling=True)
    st.caption("Si el sitio bloquea iframes, usá ‘Extraer tabla (vivo)’ o ‘Usar HTML subido’.")
    st.link_button("Abrir CRONOTRIGO en pestaña nueva", CRONOTRIGO_URL)

elif modo_crono == "Extraer tabla (vivo)":
    with st.spinner("Consultando CRONOTRIGO…"):
        try:
            html_text = fetch_cronotrigo_html()
            cronot_df = _scrape_cronotrigo_table(html_text)  # no se muestra
            p1, p2 = _extract_pc_from_html_text(html_text)
            pc_inicio, pc_fin = p1, p2
            st.success("Período Crítico detectado desde la página (si estaba presente).")
        except Exception as e:
            st.error(f"No pude leer CRONOTRIGO: {e}")

else:  # "Usar HTML subido"
    if cronot_html_file is not None:
        try:
            html_text = cronot_html_file.read().decode("utf-8", errors="ignore")
            cronot_df = _scrape_cronotrigo_table(html_text)  # no se muestra
            p1, p2 = _extract_pc_from_html_text(html_text)
            pc_inicio, pc_fin = p1, p2
            st.success("HTML leído y PC detectado (si estaba presente).")
        except Exception as e:
            st.error(f"No pude procesar el HTML subido: {e}")
    else:
        st.info("Subí el archivo HTML para continuar.")

# Si no se detectó PC y el usuario habilitó manual, uso lo ingresado
if (pc_inicio is None or pc_fin is None) and pc_manual_on:
    if pc_ini_manual and pc_fin_manual:
        pc_inicio = pd.to_datetime(pc_ini_manual)
        pc_fin = pd.to_datetime(pc_fin_manual)

# ================== PREDWEEM: Serie base + histórico opcional ==================
st.subheader("Serie PREDWEEM")
pred_vis_main = None   # serie base (según modo)
pred_vis_hist = None   # serie histórica opcional

try:
    if modo_pred == "CSV público":
        df_meteo = load_public_csv()
        pred_vis_main = run_predweem_simple(df_meteo)
        st.success(f"Base: PREDWEEM con meteo pública: {len(pred_vis_main)} días.")
    elif modo_pred == "Subir archivo (EMERREL/EMERAC)":
        if pred_file:
            pred_vis_main = run_predweem_from_file(pred_file)
            st.success(f"Base: archivo cargado ({len(pred_vis_main)} días).")
        else:
            st.info("Subí el archivo base con Fecha y EMERREL/EMERAC.")
    else:  # API MeteoBahía
        if meteo_url and meteo_url.strip():
            xml_text = fetch_meteobahia_xml(meteo_url.strip())
            df_meteo = parse_meteobahia_xml(xml_text)
            pred_vis_main = run_predweem_simple(df_meteo)
            st.success(f"Base: PREDWEEM con API MeteoBahía: {len(pred_vis_main)} días.")
        else:
            st.info("Ingresá la URL del XML de MeteoBahía.")
except Exception as e:
    st.error(f"No se pudo generar la serie base de PREDWEEM: {e}")

# Histórico opcional (para comparar)
if hist_file is not None:
    try:
        pred_vis_hist = run_predweem_from_file(hist_file)
        st.success(f"Histórico cargado para comparación: {len(pred_vis_hist)} días.")
    except Exception as e:
        st.error(f"No se pudo leer el histórico: {e}")

# --- Recorte de horizonte fijo ---
if pred_vis_main is not None:
    pred_vis_main = clip_horizon(pred_vis_main, HORIZ_INI, HORIZ_FIN)
    if pred_vis_main.empty:
        st.warning("No hay datos base de PREDWEEM en 01/02/2025 → 01/11/2025.")

if pred_vis_hist is not None:
    pred_vis_hist = clip_horizon(pred_vis_hist, HORIZ_INI, HORIZ_FIN)
    if pred_vis_hist.empty:
        st.warning("No hay datos del histórico en 01/02/2025 → 01/11/2025.")

# Ajustar PC al mismo horizonte
pc_inicio, pc_fin = clip_pc(pc_inicio, pc_fin, HORIZ_INI, HORIZ_FIN)

# ================== OVERLAP (PC vs EMERREL) ==================
def add_pc_shading(fig, pc_i, pc_f):
    if pc_i is not None and pc_f is not None and pc_i < pc_f:
        fig.add_vrect(x0=pc_i, x1=pc_f, fillcolor="rgba(255,0,0,0.12)",
                      line_width=0, layer="below",
                      annotation_text="Período crítico", annotation_position="top left")

def compute_overlap(pred_df: pd.DataFrame, pc_i, pc_f) -> tuple[pd.DataFrame, dict]:
    if pc_i is None or pc_f is None or pc_i >= pc_f or pred_df is None or pred_df.empty:
        return pd.DataFrame(), {}
    mask = (pred_df["Fecha"] >= pc_i) & (pred_df["Fecha"] <= pc_f)
    sub = pred_df.loc[mask, ["Fecha","EMERREL(0-1)","Nivel","MA5","EMERREL acumulado"]].copy()
    emerrel_pc = float(sub["EMERREL(0-1)"].sum()) if len(sub) else 0.0
    emerrel_total = float(pred_df["EMERREL(0-1)"].sum())
    pct_pc_sobre_total = emerrel_pc / emerrel_total if emerrel_total > 0 else np.nan
    resumen = {"% EMERREL en PC / total": pct_pc_sobre_total}
    return sub.reset_index(drop=True), resumen

# ================== Gráfico EMERREL (comparación) ==================
def colores_por_nivel(serie, pal=("Bajo","#2ca02c"), pb=("Medio","#ff7f0e"), pa=("Alto","#d62728")):
    mp = {pal[0]: pal[1], pb[0]: pb[1], pa[0]: pa[1]}
    return serie.map(mp).fillna("#808080").to_numpy()

fig_er = None
overlap_main_df = overlap_hist_df = pd.DataFrame()
overlap_main_res = overlap_hist_res = {}

if pred_vis_main is not None and len(pred_vis_main):
    # Asegurar 'Nivel'
    main_plot = pred_vis_main.copy()
    if "Nivel" not in main_plot.columns:
        main_plot["Nivel"] = np.where(main_plot["EMERREL(0-1)"] <= THR_BAJO_MEDIO, "Bajo",
                               np.where(main_plot["EMERREL(0-1)"] <= THR_MEDIO_ALTO, "Medio", "Alto"))

    # Histórico opcional
    hist_plot = None
    if pred_vis_hist is not None and len(pred_vis_hist):
        hist_plot = pred_vis_hist.copy()
        if "Nivel" not in hist_plot.columns:
            hist_plot["Nivel"] = np.where(hist_plot["EMERREL(0-1)"] <= THR_BAJO_MEDIO, "Bajo",
                                   np.where(hist_plot["EMERREL(0-1)"] <= THR_MEDIO_ALTO, "Medio", "Alto"))

    st.subheader("EMERREL diario (MA5 + sombreado PC) · Comparación")
    fig_er = go.Figure()

    # Base: barras por nivel + MA5
    colors_main = colores_por_nivel(main_plot["Nivel"])
    fig_er.add_trace(go.Scatter(x=main_plot["Fecha"], y=main_plot["MA5"], mode="lines",
                                line=dict(width=0), fill="tozeroy", fillcolor=rgba("#4169e1",0.15),
                                showlegend=False, hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=main_plot["Fecha"], y=main_plot["MA5"], mode="lines",
                                line=dict(width=2), name="Base · MA5",
                                hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Base MA5: %{y:.3f}<extra></extra>"))
    fig_er.add_bar(x=main_plot["Fecha"], y=main_plot["EMERREL(0-1)"],
                   marker=dict(color=colors_main.tolist()),
                   customdata=main_plot["Nivel"].map({"Bajo":"🟢 Bajo","Medio":"🟠 Medio","Alto":"🔴 Alto"}),
                   hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Base EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
                   name="Base · EMERREL (0-1)")

    # Histórico: líneas (para no superponer barras)
    if hist_plot is not None:
        fig_er.add_trace(go.Scatter(x=hist_plot["Fecha"], y=hist_plot["MA5"], mode="lines",
                                    line=dict(width=2, dash="dash"),
                                    name="Histórico · MA5",
                                    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Histórico MA5: %{y:.3f}<extra></extra>"))
        fig_er.add_trace(go.Scatter(x=hist_plot["Fecha"], y=hist_plot["EMERREL(0-1)"], mode="lines",
                                    line=dict(width=1),
                                    name="Histórico · EMERREL (línea)",
                                    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Histórico EMERREL: %{y:.3f}<extra></extra>"))

    # Líneas de referencia + PC
    fig_er.add_hline(y=THR_BAJO_MEDIO, line_dash="dot", opacity=0.6, annotation_text=f"Bajo ≤ {THR_BAJO_MEDIO:.3f}")
    fig_er.add_hline(y=THR_MEDIO_ALTO, line_dash="dot", opacity=0.6, annotation_text=f"Medio ≤ {THR_MEDIO_ALTO:.3f}")
    add_pc_shading(fig_er, pc_inicio, pc_fin)
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)", hovermode="x unified",
                         height=560, legend_title="Series")
    st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    # --- OVERLAP: mostrar SOLO "% en PC / Total" para cada serie disponible ---
    st.subheader("Overlap CRONOTRIGO × PREDWEEM — % en PC / Total")
    cols = st.columns(2 if pred_vis_hist is not None and len(pred_vis_hist) else 1)

    # Base
    if pc_inicio is not None and pc_fin is not None and pc_inicio < pc_fin:
        overlap_main_df, overlap_main_res = compute_overlap(main_plot, pc_inicio, pc_fin)
        pct_main = overlap_main_res.get("% EMERREL en PC / total", np.nan)
        cols[0].metric("Base · % en PC / Total", f"{pct_main:.0%}" if pd.notna(pct_main) else "—")
    else:
        cols[0].warning("PC inválido o fuera del horizonte.")

    # Histórico
    if pred_vis_hist is not None and len(pred_vis_hist):
        if pc_inicio is not None and pc_fin is not None and pc_inicio < pc_fin:
            overlap_hist_df, overlap_hist_res = compute_overlap(hist_plot, pc_inicio, pc_fin)
            pct_hist = overlap_hist_res.get("% EMERREL en PC / total", np.nan)
            cols[1].metric("Histórico · % en PC / Total", f"{pct_hist:.0%}" if pd.notna(pct_hist) else "—")
        else:
            cols[1].warning("PC inválido o fuera del horizonte.")

    # Info PC aplicado
    if pc_inicio is not None and pc_fin is not None and pc_inicio < pc_fin:
        st.caption(f"Período crítico aplicado: {pc_inicio.date().strftime('%d/%m/%Y')} → {pc_fin.date().strftime('%d/%m/%Y')}  "
                   f"· Días: {(pc_fin - pc_inicio).days + 1}")
else:
    st.info("Aún no hay serie base disponible para graficar.")

# ================== Descargas ==================
st.subheader("Descargas")
cols = st.columns(4)

# (Opcional) CSV de CRONOTRIGO para quien lo necesite (aunque no se muestra en pantalla)
if cronot_df is not None and not cronot_df.empty:
    buf_ct = io.StringIO(); cronot_df.to_csv(buf_ct, index=False)
    cols[0].download_button("⬇ CRONOTRIGO (tabla, CSV)", data=buf_ct.getvalue(),
                            file_name="cronotrigo_tabla_riesgos.csv", mime="text/csv")

# Serie base
if pred_vis_main is not None and not pred_vis_main.empty:
    buf_p = io.StringIO(); pred_vis_main.to_csv(buf_p, index=False)
    cols[1].download_button("⬇ PREDWEEM base (CSV)", data=buf_p.getvalue(),
                            file_name="predweem_base_2025_clip.csv", mime="text/csv")

# Serie histórica (si existe)
if pred_vis_hist is not None and not pred_vis_hist.empty:
    buf_h = io.StringIO(); pred_vis_hist.to_csv(buf_h, index=False)
    cols[2].download_button("⬇ PREDWEEM histórico (CSV)", data=buf_h.getvalue(),
                            file_name="predweem_historico_2025_clip.csv", mime="text/csv")

# Overlaps (si existen)
def fig_to_html_bytes(fig):
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

zip_ready = any([
    cronot_df is not None and not (isinstance(cronot_df, pd.DataFrame) and cronot_df.empty),
    pred_vis_main is not None and not pred_vis_main.empty,
    pred_vis_hist is not None and not pred_vis_hist.empty
])

if zip_ready:
    with io.BytesIO() as mem:
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if cronot_df is not None and not (isinstance(cronot_df, pd.DataFrame) and cronot_df.empty):
                _b = io.StringIO(); cronot_df.to_csv(_b, index=False)
                zf.writestr("cronotrigo_tabla_riesgos.csv", _b.getvalue())
            if pred_vis_main is not None and not pred_vis_main.empty:
                _b = io.StringIO(); pred_vis_main.to_csv(_b, index=False)
                zf.writestr("predweem_base_2025_clip.csv", _b.getvalue())
            if pred_vis_hist is not None and not pred_vis_hist.empty:
                _b = io.StringIO(); pred_vis_hist.to_csv(_b, index=False)
                zf.writestr("predweem_historico_2025_clip.csv", _b.getvalue())
            # Overlaps por serie (si se calculó)
            if 'overlap_main_df' in locals() and isinstance(overlap_main_df, pd.DataFrame) and not overlap_main_df.empty:
                _b = io.StringIO(); overlap_main_df.to_csv(_b, index=False)
                zf.writestr("overlap_predweem_base_pc_2025.csv", _b.getvalue())
            if 'overlap_hist_df' in locals() and isinstance(overlap_hist_df, pd.DataFrame) and not overlap_hist_df.empty:
                _b = io.StringIO(); overlap_hist_df.to_csv(_b, index=False)
                zf.writestr("overlap_predweem_historico_pc_2025.csv", _b.getvalue())
            # Gráfico EMERREL
            if 'fig_er' in locals() and fig_er is not None:
                zf.writestr("grafico_emerrel_comparacion.html", fig_to_html_bytes(fig_er))
        mem.seek(0)
        cols[3].download_button("⬇ Descargar TODO (ZIP)", data=mem.read(),
                                file_name="cronotrigo_predweem_paquete_2025_comparacion.zip", mime="application/zip")

