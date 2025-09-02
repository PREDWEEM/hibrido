# -*- coding: utf-8 -*-
# app_cronotrigo_predweem.py
#
# Streamlit: Carga imagen de CRONOTrigo -> OCR -> extracción de fechas
# Integración con PREDWEEM (subir CSV/XLSX) o correr PREDWEEM simple (pesos remotos + meteo pública)
# Visualización: EMERREL diario (MA5 + sombreado + vrect Período Crítico), EMEAC (%), línea de tiempo fenológica
#
# Requisitos (requirements.txt sugerido):
# streamlit pandas numpy plotly Pillow rapidocr-onnxruntime pytesseract

import io
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# =================== LOOK & FEEL (lockdown suave) ===================
st.set_page_config(page_title="CRONOTrigo + PREDWEEM · Integración", layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("CRONOTrigo + PREDWEEM · Integración desde imagen")

# =================== UTILIDADES GENERALES ===================
def _norm_col(df, aliases):
    for a in aliases:
        if a in df.columns: return a
    return None

DATE_PAT = re.compile(
    r"(?P<d>\d{1,2})[/-](?P<m>\d{1,2})(?:[/-](?P<y>\d{2,4}))?(?:\s*[±\+/-]\s*(?P<u>\d+))?",
    re.UNICODE
)
NUM_PAT = re.compile(r"(-?\d+(?:[.,]\d+)?)")

def _to_datetime_safe(s, default_year=None):
    if s is None: return (pd.NaT, float("nan"))
    s = str(s).strip()
    m = DATE_PAT.search(s)
    if not m:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return (dt, float("nan"))
    d, mth, y = int(m.group("d")), int(m.group("m")), m.group("y")
    if y is None:
        y = default_year if default_year else pd.Timestamp.now().year
    y = int(y) if len(str(y)) == 4 else 2000 + int(y)
    unc = float(m.group("u")) if m.group("u") else float("nan")
    try:
        return (pd.Timestamp(year=y, month=mth, day=d), unc)
    except Exception:
        return (pd.NaT, unc)

def _num(s, pct=False):
    if s is None: return float("nan")
    m = NUM_PAT.search(str(s).replace(",", "."))
    if not m: return float("nan")
    v = float(m.group(1))
    return v/100.0 if pct and v > 1 else v

def _find_line(block, *needles):
    for ln in block.splitlines():
        L = ln.lower()
        if all(n.lower() in L for n in needles):
            return ln.strip()
    return ""

# =================== OCR: RapidOCR (preferido) -> pytesseract (fallback) ===================
@st.cache_resource(show_spinner=False)
def _get_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception:
        return None

def ocr_text_from_image(img_bytes: bytes) -> str:
    # 1) RapidOCR
    try:
        ocr = _get_rapidocr()
        if ocr:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            res, _ = ocr(img)
            if res:
                lines = [r[1] for r in res if isinstance(r, (list, tuple)) and len(r) >= 2]
                txt = "\n".join(lines)
                if len(txt.strip()) >= 10:
                    return txt
    except Exception:
        pass
    # 2) pytesseract (si está instalado el binario en el sistema)
    try:
        import pytesseract
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        try:
            txt = pytesseract.image_to_string(img, lang="spa")
        except Exception:
            txt = pytesseract.image_to_string(img)
        if len(txt.strip()) >= 10:
            return txt
    except Exception:
        pass
    return ""

# =================== PARSEO TEXTO CRONOTRIGO DESDE OCR ===================
def parse_cronotrigo_text(txt: str):
    T = re.sub(r"[ \t]+", " ", txt)
    T = re.sub(r"[–—]+", "-", T)

    labels = {
        "Siembra":      ["siembra"],
        "Emergencia":   ["emergencia"],
        "Primer_nudo":  ["primer nudo", "z31", "nudo visible", "encañazón", "primer-nudo", "primer-nudo"],
        "Espigazon":    ["espig", "espigazón", "espiga"],
        "Antesis":      ["antesis", "floración", "floracion"],
        "Madurez":      ["madurez", "madurez fisiologica", "madurez fisiológica"],
    }

    filas, years_seen = [], []
    for k, keys in labels.items():
        ln = ""
        for kw in keys:
            ln = _find_line(T, kw)
            if ln: break
        if ln:
            dt, unc = _to_datetime_safe(ln)
            if pd.notna(dt): years_seen.append(dt.year)
            filas.append({"Estadio": k, "Fecha": dt, "±d": unc})

    default_year = max(years_seen) if years_seen else pd.Timestamp.now().year

    # Período crítico
    ln_pc = _find_line(T, "período crítico") or _find_line(T, "periodo critico")
    pc_ini, pc_fin = pd.NaT, pd.NaT
    if ln_pc:
        matches = DATE_PAT.findall(ln_pc)
        if len(matches) >= 2:
            def make_date(m):
                dd, mm, yy, uu = m
                frag = f"{dd}/{mm}" + (f"/{yy}" if yy else "")
                return _to_datetime_safe(frag, default_year)[0]
            pc_ini = make_date(matches[0])
            pc_fin = make_date(matches[1])
    else:
        ln_ini = _find_line(T, "inicio", "crítico") or _find_line(T, "inicio", "critico")
        ln_fin = _find_line(T, "fin", "crítico") or _find_line(T, "fin", "critico")
        if ln_ini: pc_ini, _ = _to_datetime_safe(ln_ini, default_year)
        if ln_fin: pc_fin, _ = _to_datetime_safe(ln_fin, default_year)

    # Agua/TT si aparecen
    ln_sw = _find_line(T, "agua", "suelo") or _find_line(T, "suelo", "agua")
    agua_frac = _num(ln_sw, pct=True) if ln_sw else float("nan")
    ln_tt = _find_line(T, "tt") or _find_line(T, "térmico") or _find_line(T, "termico")
    tt_val = _num(ln_tt) if ln_tt else float("nan")

    estados = pd.DataFrame(filas)
    if estados.empty:
        raise ValueError("No se reconocieron estadios en la imagen.")

    # Normalizar orden y completar faltantes con NaT
    needed = ["Siembra","Emergencia","Primer_nudo","Espigazon","Antesis","Madurez"]
    for n in needed:
        if not (estados["Estadio"] == n).any():
            estados.loc[len(estados)] = {"Estadio": n, "Fecha": pd.NaT, "±d": float("nan")}
    estados = estados.sort_values("Fecha").reset_index(drop=True)

    key_dates = {r["Estadio"]: r["Fecha"] for _, r in estados.iterrows()}
    ventanas = pd.DataFrame({
        "Ventana": ["Período crítico"],
        "Inicio": [pc_ini],
        "Fin": [pc_fin],
        "Duración (d)": [(pc_fin - pc_ini).days if (pd.notna(pc_ini) and pd.notna(pc_fin)) else float("nan")]
    })
    meta = {"AguaSuelo_frac": agua_frac, "TT_aprox": tt_val}
    return estados, ventanas, key_dates, meta, T

# =================== PREDWEEM: Carga o Correr Simple ===================
THR_BAJO_MEDIO = 0.020
THR_MEDIO_ALTO = 0.079
COLOR_MAP = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
COLOR_FALLBACK = "#808080"

EMEAC_MIN_DEN = 1.8
EMEAC_ADJ_DEN = 2.1
EMEAC_MAX_DEN = 2.5

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
                df = df.sort_values("Fecha").reset_index(drop=True)
                return df
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
    # Normalizar columnas mínimas
    col_f = _norm_col(df, ["Fecha","date","Day","dia"])
    if not col_f:
        raise ValueError("El archivo de PREDWEEM debe incluir una columna de Fecha.")
    df["Fecha"] = pd.to_datetime(df[col_f], dayfirst=True, errors="coerce")
    # EMERREL/EMERAC
    col_e = _norm_col(df, ["EMERREL","EMERAC","EmergenciaRel","emerrel","emerac"])
    if not col_e:
        raise ValueError("No se encontró EMERREL/EMERAC en el archivo de PREDWEEM.")
    ser = pd.to_numeric(df[col_e], errors="coerce").fillna(0.0)
    if ser.max() > 1.0001: ser = ser / ser.max()
    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df["Fecha"]),
        "EMERREL(0-1)": ser
    }).sort_values("Fecha")
    out["EMERREL acumulado"] = out["EMERREL(0-1)"].cumsum().clip(upper=1.0)
    out["MA5"] = out["EMERREL(0-1)"].rolling(5, min_periods=1).mean()
    # si tiene Julian_days, conservar
    if "Julian_days" in df.columns:
        out["Julian_days"] = pd.to_numeric(df["Julian_days"], errors="coerce")
    return out.reset_index(drop=True)

# =================== SIDEBAR: Entradas ===================
with st.sidebar:
    st.header("Entradas")
    st.markdown("**1) Imagen de CRONOTrigo**")
    file_img = st.file_uploader("Subí PNG/JPG", type=["png","jpg","jpeg"])

    st.markdown("---")
    st.markdown("**2) PREDWEEM**")
    modo_pred = st.radio(
        "¿Cómo integrar PREDWEEM?",
        ["Subir archivo (EMERREL/EMERAC)", "Correr simple con meteo pública"],
        index=0
    )
    pred_file = None
    pred_df = None
    if modo_pred == "Subir archivo (EMERREL/EMERAC)":
        pred_file = st.file_uploader("CSV/XLSX PREDWEEM diario", type=["csv","xlsx"], key="pred_up")
    else:
        st.caption("Usará pesos remotos y meteo pública (gh-pages).")

# =================== PROCESO: CRONOTrigo desde imagen ===================
estados = ventanas = key_dates = meta = None
ocr_txt = ""

col_left, col_right = st.columns([1, 2], vertical_alignment="top")

with col_left:
    st.subheader("Imagen de CRONOTrigo")
    if file_img:
        img_bytes = file_img.read()
        st.image(img_bytes, use_container_width=True, caption=file_img.name)
        with st.spinner("Extrayendo texto (OCR)…"):
            ocr_txt = ocr_text_from_image(img_bytes)
        if not ocr_txt:
            st.error("No se pudo leer texto de la imagen. Probá con una captura más nítida.")
    else:
        st.info("Subí una imagen para continuar.")

with col_right:
    st.subheader("Resultados extraídos")
    if ocr_txt:
        try:
            estados, ventanas, key_dates, meta, T = parse_cronotrigo_text(ocr_txt)
            with st.expander("Texto detectado (OCR)"):
                st.code(T[:4000])
            st.markdown("**Estadios**")
            st.dataframe(estados, use_container_width=True)
            st.markdown("**Período crítico**")
            st.table(ventanas)
            if isinstance(meta, dict):
                if not np.isnan(meta.get("AguaSuelo_frac", np.nan)):
                    st.info(f"Agua en Suelo (extraída): ~{meta['AguaSuelo_frac']*100:.0f}%")
                if not np.isnan(meta.get("TT_aprox", np.nan)):
                    st.info(f"Tiempo térmico (aprox.): {meta['TT_aprox']:.0f} °C·día")
        except Exception as e:
            st.error(f"No se pudo interpretar la imagen: {e}")
            estados = ventanas = key_dates = None

# =================== PREDWEEM: Carga o corrida simple ===================
st.subheader("Serie PREDWEEM")
pred_vis = None
if modo_pred == "Subir archivo (EMERREL/EMERAC)":
    if pred_file:
        try:
            pred_vis = run_predweem_from_file(pred_file)
            st.success(f"Serie cargada: {len(pred_vis)} días.")
        except Exception as e:
            st.error(f"No se pudo leer PREDWEEM: {e}")
    else:
        st.info("Subí un archivo con Fecha y EMERREL/EMERAC.")
else:
    try:
        df_meteo = load_public_csv()
        pred_vis = run_predweem_simple(df_meteo)
        st.success(f"PREDWEEM corrido con meteo pública: {len(pred_vis)} días.")
    except Exception as e:
        st.error(f"No se pudo correr PREDWEEM simple: {e}")

# =================== GRÁFICOS E INTEGRACIÓN ===================
def colores_por_nivel(serie):
    return serie.map(COLOR_MAP).fillna(COLOR_FALLBACK).to_numpy()

def add_periodo_critico(fig, ventanas):
    try:
        pc = ventanas.iloc[0]
        if pd.notna(pc["Inicio"]) and pd.notna(pc["Fin"]) and (pc["Inicio"] < pc["Fin"]):
            fig.add_vrect(x0=pc["Inicio"], x1=pc["Fin"],
                          fillcolor="rgba(255,0,0,0.12)", line_width=0,
                          layer="below", annotation_text="Período crítico",
                          annotation_position="top left")
    except Exception:
        pass

def integrar_metricas(pred_df, key_dates, ventanas):
    z31 = key_dates.get("Primer_nudo", pd.NaT) if key_dates else pd.NaT
    esp = key_dates.get("Espigazon", pd.NaT) if key_dates else pd.NaT
    def pct_before(corte):
        if pd.isna(corte): return float("nan")
        m = pred_df.loc[pred_df["Fecha"] <= corte, "EMERREL acumulado"]
        return float(m.max()) if len(m) else float("nan")
    pct_antes_z31 = pct_before(z31)
    pct_antes_esp = pct_before(esp)
    if ventanas is not None and len(ventanas):
        pc = ventanas.iloc[0]
        if pd.notna(pc["Inicio"]) and pd.notna(pc["Fin"]) and (pc["Inicio"] < pc["Fin"]):
            sub = pred_df[(pred_df["Fecha"]>=pc["Inicio"]) & (pred_df["Fecha"]<=pc["Fin"])]
            pct_en_pc = float(sub["EMERREL(0-1)"].sum())
            pct_en_pc = min(pct_en_pc, 1.0)
        else:
            pct_en_pc = float("nan")
    else:
        pct_en_pc = float("nan")
    return pd.DataFrame([{
        "Emerg. rel. antes Z31": pct_antes_z31,
        "Emerg. rel. antes Espigazón": pct_antes_esp,
        "Emerg. rel. en Período crítico": pct_en_pc
    }])

if pred_vis is not None and len(pred_vis):
    # EMERREL diario (barras + MA5 + sombreado MA5)
    st.subheader("EMERREL diario (con MA5)")
    pred_plot = pred_vis.copy()
    # Clasificación de niveles si vino de archivo sin 'Nivel'
    if "Nivel" not in pred_plot.columns:
        th1, th2 = THR_BAJO_MEDIO, THR_MEDIO_ALTO
        pred_plot["Nivel"] = np.where(pred_plot["EMERREL(0-1)"] <= th1, "Bajo",
                               np.where(pred_plot["EMERREL(0-1)"] <= th2, "Medio", "Alto"))
    colors = colores_por_nivel(pred_plot["Nivel"])
    fig_er = go.Figure()
    # Sombreado MA5 primero (queda debajo)
    fig_er.add_trace(go.Scatter(
        x=pred_plot["Fecha"], y=pred_plot["MA5"],
        mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
        showlegend=False, hoverinfo="skip"
    ))
    # Línea MA5
    fig_er.add_trace(go.Scatter(
        x=pred_plot["Fecha"], y=pred_plot["MA5"],
        mode="lines", line=dict(width=2),
        name="Media móvil (5d)",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
    ))
    # Barras EMERREL
    fig_er.add_bar(
        x=pred_plot["Fecha"], y=pred_plot["EMERREL(0-1)"],
        marker=dict(color=colors.tolist()),
        customdata=pred_plot["Nivel"].map({"Bajo":"🟢 Bajo","Medio":"🟠 Medio","Alto":"🔴 Alto"}),
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
        name="EMERREL (0-1)"
    )
    # Líneas de referencia
    low_thr, med_thr = float(THR_BAJO_MEDIO), float(THR_MEDIO_ALTO)
    fig_er.add_hline(y=low_thr, line_dash="dot", annotation_text=f"Bajo (≤ {low_thr:.3f})", opacity=0.6)
    fig_er.add_hline(y=med_thr, line_dash="dot", annotation_text=f"Medio (≤ {med_thr:.3f})", opacity=0.6)
    add_periodo_critico(fig_er, ventanas)  # sombrear PC si existe
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
                         hovermode="x unified", height=520, legend_title="Referencias")
    st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    # EMEAC (%)
    st.subheader("EMEAC (%)")
    emeac = pd.DataFrame({
        "Fecha": pred_plot["Fecha"],
        "EMEAC_min": (pred_plot["EMERREL acumulado"] / EMEAC_MIN_DEN * 100).clip(0, 100),
        "EMEAC_adj": (pred_plot["EMERREL acumulado"] / EMEAC_ADJ_DEN * 100).clip(0, 100),
        "EMEAC_max": (pred_plot["EMERREL acumulado"] / EMEAC_MAX_DEN * 100).clip(0, 100),
    })
    fig_ac = go.Figure()
    fig_ac.add_trace(go.Scatter(x=emeac["Fecha"], y=emeac["EMEAC_max"], mode="lines",
                                line=dict(width=0), name="Máximo"))
    fig_ac.add_trace(go.Scatter(x=emeac["Fecha"], y=emeac["EMEAC_min"], mode="lines",
                                line=dict(width=0), fill="tonexty", name="Mínimo"))
    fig_ac.add_trace(go.Scatter(x=emeac["Fecha"], y=emeac["EMEAC_adj"], mode="lines",
                                line=dict(width=2.5), name=f"Ajustable (/{EMEAC_ADJ_DEN:.2f})"))
    for nivel in [25, 50, 75, 90]:
        try:
            fig_ac.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
        except Exception:
            pass
    add_periodo_critico(fig_ac, ventanas)
    fig_ac.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)",
                         hovermode="x unified", height=480, legend_title="Referencias")
    st.plotly_chart(fig_ac, use_container_width=True, theme="streamlit")

    # Métricas integradas
    if key_dates is not None:
        st.subheader("Métricas integradas (CRONOTrigo + PREDWEEM)")
        resumen = integrar_metricas(pred_plot, key_dates, ventanas)
        st.dataframe(resumen.style.format({c:"{:.0%}" for c in resumen.columns}),
                     use_container_width=True)

# =================== TIMELINE FENOLÓGICA (CRONOTrigo) ===================
if estados is not None:
    st.subheader("CRONOTrigo · Línea de tiempo fenológica")
    fig_tl = go.Figure()
    # Período crítico
    if ventanas is not None and len(ventanas):
        pc = ventanas.iloc[0]
        if pd.notna(pc["Inicio"]) and pd.notna(pc["Fin"]) and pc["Inicio"] < pc["Fin"]:
            fig_tl.add_vrect(x0=pc["Inicio"], x1=pc["Fin"],
                             fillcolor="rgba(255,0,0,0.12)", line_width=0,
                             layer="below", annotation_text="Período crítico",
                             annotation_position="top left")
    # vlines por estadio
    y = 1
    for _, r in estados.sort_values("Fecha").iterrows():
        if pd.isna(r["Fecha"]): continue
        fig_tl.add_vline(x=r["Fecha"], line=dict(color="rgba(0,0,0,0.5)", width=1))
        fig_tl.add_annotation(x=r["Fecha"], y=y, text=r["Estadio"],
                              showarrow=True, arrowhead=2, yshift=30)
    # base
    fechas = estados["Fecha"].dropna()
    if len(fechas):
        fi, ff = fechas.min() - pd.Timedelta(days=10), fechas.max() + pd.Timedelta(days=10)
        fig_tl.add_scatter(x=[fi, ff], y=[y, y], mode="lines", line=dict(width=0), showlegend=False)
    fig_tl.update_yaxes(visible=False, range=[0, 2])
    fig_tl.update_layout(xaxis_title="Fecha", hovermode="x unified", height=300)
    st.plotly_chart(fig_tl, use_container_width=True, theme="streamlit")

# =================== DESCARGAS ===================
st.subheader("Descargas")
cols = st.columns(3)
if estados is not None:
    buf = io.StringIO()
    estados_out = estados.copy()
    estados_out["Fecha"] = estados_out["Fecha"].dt.strftime("%Y-%m-%d")
    estados_out.to_csv(buf, index=False)
    cols[0].download_button("⬇ Estadios (CSV)", data=buf.getvalue(),
                            file_name="cronotrigo_estadios_ocr.csv", mime="text/csv")

if ventanas is not None:
    buf2 = io.StringIO()
    ventanas_out = ventanas.copy()
    if "Duración (d)" in ventanas_out.columns:
        ventanas_out.rename(columns={"Duración (d)":"Duracion_d"}, inplace=True)
    ventanas_out.to_csv(buf2, index=False)
    cols[1].download_button("⬇ Período crítico (CSV)", data=buf2.getvalue(),
                            file_name="cronotrigo_periodo_critico.csv", mime="text/csv")

if pred_vis is not None:
    buf3 = io.StringIO()
    pred_vis.to_csv(buf3, index=False)
    cols[2].download_button("⬇ Serie PREDWEEM (CSV)", data=buf3.getvalue(),
                            file_name="predweem_serie.csv", mime="text/csv")

st.caption("Tip: si el OCR no reconoce bien, subí una imagen con buena resolución y sin recortes. Si necesitás, puedo agregar un formulario de carga manual como fallback.")

