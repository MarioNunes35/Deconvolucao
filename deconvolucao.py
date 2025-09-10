
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares

st.set_page_config(page_title="Deconvolução Espectral — Gaussian/Lorentzian/Pseudo-Voigt", layout="wide")

# ========================= Helpers =========================
def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(int(niter)):
        W = sp.diags(w, 0, shape=(L, L))
        Z = W + lam * D @ D.T
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)

def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def lorentzian(x, A, x0, gamma):
    return A * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)

def pseudo_voigt(x, A, x0, sigma, gamma, eta):
    # eta in [0,1]
    eta = np.clip(eta, 0.0, 1.0)
    return (1 - eta) * gaussian(x, A, x0, sigma) + eta * lorentzian(x, A, x0, gamma)

def fwhm_from_sigma(sigma):
    return 2.354820045 * sigma

def sigma_from_fwhm(fwhm):
    return fwhm / 2.354820045

def gamma_from_fwhm(fwhm):
    return fwhm / 2.0

def area_gaussian(A, sigma):
    return A * sigma * np.sqrt(2 * np.pi)

def area_lorentzian(A, gamma):
    return A * np.pi * gamma

def detect_orientation(y):
    # Decide if peaks are "up" or "down" comparing spans.
    up_span = np.max(y) - np.median(y)
    down_span = np.median(y) - np.min(y)
    return "down" if down_span > up_span else "up"

def build_initial_params(x, y_fit_target, centers, profile, fwhm_guess, widths_free, eta_fixed, eta_value):
    params = []
    for c in centers:
        idx = (np.abs(x - c)).argmin()
        A0 = max(y_fit_target[idx] - np.median(y_fit_target), 1e-9)
        if profile == "Gaussian":
            sigma0 = sigma_from_fwhm(fwhm_guess)
            if widths_free:
                params.extend([A0, c, sigma0])
            else:
                params.extend([A0, c])  # width fixed later via mapping
        elif profile == "Lorentzian":
            gamma0 = gamma_from_fwhm(fwhm_guess)
            if widths_free:
                params.extend([A0, c, gamma0])
            else:
                params.extend([A0, c])
        elif profile == "Pseudo-Voigt":
            sigma0 = sigma_from_fwhm(fwhm_guess)
            gamma0 = gamma_from_fwhm(fwhm_guess)
            if widths_free and not eta_fixed:
                params.extend([A0, c, sigma0, gamma0, eta_value])
            elif widths_free and eta_fixed:
                params.extend([A0, c, sigma0, gamma0])
            elif (not widths_free) and not eta_fixed:
                params.extend([A0, c, eta_value])
            else:
                params.extend([A0, c])
    # global fixed width when widths_free=False
    global_width = None
    if not widths_free:
        global_width = fwhm_guess
    return np.array(params, dtype=float), global_width

def model_sum(x, params, profile, centers, widths_free, global_width, eta_fixed, eta_value):
    y = np.zeros_like(x, dtype=float)
    i = 0
    for c in centers:
        if profile == "Gaussian":
            if widths_free:
                A, x0, sigma = params[i:i+3]; i += 3
            else:
                A, x0 = params[i:i+2]; i += 2
                sigma = sigma_from_fwhm(global_width)
            y += gaussian(x, A, x0, sigma)
        elif profile == "Lorentzian":
            if widths_free:
                A, x0, gamma = params[i:i+3]; i += 3
            else:
                A, x0 = params[i:i+2]; i += 2
                gamma = gamma_from_fwhm(global_width)
            y += lorentzian(x, A, x0, gamma)
        elif profile == "Pseudo-Voigt":
            if widths_free and not eta_fixed:
                A, x0, sigma, gamma, eta = params[i:i+5]; i += 5
            elif widths_free and eta_fixed:
                A, x0, sigma, gamma = params[i:i+4]; i += 4
                eta = eta_value
            elif (not widths_free) and not eta_fixed:
                A, x0, eta = params[i:i+3]; i += 3
                sigma = sigma_from_fwhm(global_width)
                gamma = gamma_from_fwhm(global_width)
            else:  # both fixed
                A, x0 = params[i:i+2]; i += 2
                sigma = sigma_from_fwhm(global_width)
                gamma = gamma_from_fwhm(global_width)
                eta = eta_value
            y += pseudo_voigt(x, A, x0, sigma, gamma, eta)
    return y

def build_bounds(centers, profile, allow_shift, max_shift, nonneg, widths_free, eta_fixed):
    lb, ub = [], []
    for c in centers:
        if profile in ("Gaussian", "Lorentzian"):
            # A, x0, [width]
            lb += [0.0 if nonneg else -np.inf, c - max_shift if allow_shift else c]
            ub += [np.inf,            c + max_shift if allow_shift else c]
            if widths_free:
                lb += [1e-9]; ub += [np.inf]
        elif profile == "Pseudo-Voigt":
            # A, x0, [sigma, gamma, eta]
            lb += [0.0 if nonneg else -np.inf, c - max_shift if allow_shift else c]
            ub += [np.inf,            c + max_shift if allow_shift else c]
            if widths_free:
                lb += [1e-9, 1e-9]
                ub += [np.inf, np.inf]
            if not eta_fixed:
                lb += [0.0]; ub += [1.0]
    return np.array(lb, dtype=float), np.array(ub, dtype=float)

def fit_model(x, y_proc, centers, profile, fwhm_guess, allow_shift, max_shift,
              nonneg, widths_free, invert_peaks, eta_fixed, eta_value):
    y_fit_target = -y_proc if invert_peaks else y_proc
    p0, global_width = build_initial_params(
        x, y_fit_target, centers, profile, fwhm_guess, widths_free, eta_fixed, eta_value
    )
    lb, ub = build_bounds(centers, profile, allow_shift, max_shift, nonneg, widths_free, eta_fixed)

    def residuals(p):
        return model_sum(x, p, profile, centers, widths_free, global_width, eta_fixed, eta_value) - y_fit_target

    res = least_squares(residuals, p0, bounds=(lb, ub))
    p_opt = res.x
    y_fit = model_sum(x, p_opt, profile, centers, widths_free, global_width, eta_fixed, eta_value)
    return p_opt, y_fit, res

def params_table(x, params, profile, centers, widths_free, global_width, eta_fixed, eta_value):
    rows = []
    i = 0
    idx = 1
    for c in centers:
        if profile == "Gaussian":
            if widths_free:
                A, x0, sigma = params[i:i+3]; i += 3
            else:
                A, x0 = params[i:i+2]; i += 2
                sigma = sigma_from_fwhm(global_width)
            fwhm = fwhm_from_sigma(sigma)
            area = area_gaussian(A, sigma)
            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm=fwhm, area=area, profile="Gaussian"))
        elif profile == "Lorentzian":
            if widths_free:
                A, x0, gamma = params[i:i+3]; i += 3
            else:
                A, x0 = params[i:i+2]; i += 2
                gamma = gamma_from_fwhm(global_width)
            fwhm = 2 * gamma
            area = area_lorentzian(A, gamma)
            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm=fwhm, area=area, profile="Lorentzian"))
        elif profile == "Pseudo-Voigt":
            if widths_free and not eta_fixed:
                A, x0, sigma, gamma, eta = params[i:i+5]; i += 5
            elif widths_free and eta_fixed:
                A, x0, sigma, gamma = params[i:i+4]; i += 4
                eta = eta_value
            elif (not widths_free) and not eta_fixed:
                A, x0, eta = params[i:i+3]; i += 3
                sigma = sigma_from_fwhm(global_width)
                gamma = gamma_from_fwhm(global_width)
            else:
                A, x0 = params[i:i+2]; i += 2
                sigma = sigma_from_fwhm(global_width)
                gamma = gamma_from_fwhm(global_width)
                eta = eta_value
            fwhm_g = fwhm_from_sigma(sigma)
            fwhm_l = 2 * gamma
            area = (1 - eta) * area_gaussian(A, sigma) + eta * area_lorentzian(A, gamma)
            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm_gaussian=fwhm_g, fwhm_lorentzian=fwhm_l, eta=eta, area_est=area, profile="Pseudo-Voigt"))
        idx += 1
    return pd.DataFrame(rows)

def component_curves(x, params, profile, centers, widths_free, global_width, eta_fixed, eta_value):
    # returns list of (label, y_i)
    curves = []
    i = 0
    for c in centers:
        if profile == "Gaussian":
            if widths_free:
                A, x0, sigma = params[i:i+3]; i += 3
            else:
                A, x0 = params[i:i+2]; i += 2
                sigma = sigma_from_fwhm(global_width)
            yi = gaussian(x, A, x0, sigma)
            curves.append((f"Peak @ {x0:.2f}", yi))
        elif profile == "Lorentzian":
            if widths_free:
                A, x0, gamma = params[i:i+3]; i += 3
            else:
                A, x0 = params[i:i+2]; i += 2
                gamma = gamma_from_fwhm(global_width)
            yi = lorentzian(x, A, x0, gamma)
            curves.append((f"Peak @ {x0:.2f}", yi))
        elif profile == "Pseudo-Voigt":
            if widths_free and not eta_fixed:
                A, x0, sigma, gamma, eta = params[i:i+5]; i += 5
            elif widths_free and eta_fixed:
                A, x0, sigma, gamma = params[i:i+4]; i += 4
                eta = eta_value
            elif (not widths_free) and not eta_fixed:
                A, x0, eta = params[i:i+3]; i += 3
                sigma = sigma_from_fwhm(global_width)
                gamma = gamma_from_fwhm(global_width)
            else:
                A, x0 = params[i:i+2]; i += 2
                sigma = sigma_from_fwhm(global_width)
                gamma = gamma_from_fwhm(global_width)
                eta = eta_value
            yi = pseudo_voigt(x, A, x0, sigma, gamma, eta)
            curves.append((f"Peak @ {x0:.2f} (η={eta:.2f})", yi))
    return curves

def r2_rmse(y_true, y_pred):
    resid = y_true - y_pred
    sse = np.sum(resid**2)
    sst = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    r2 = 1 - sse/sst
    rmse = np.sqrt(np.mean(resid**2))
    return r2, rmse

# ========================= UI =========================
st.title("🔬 Deconvolução de Espectros (Gaussian / Lorentzian / Pseudo-Voigt)")

with st.expander("Como usar", expanded=False):
    st.markdown("""
1. Carregue um **CSV/XLSX** com duas colunas: **X** (ex.: número de onda) e **Y** (intensidade).
2. Ajuste **suavização** e **linha base**.
3. Defina os **centros** manualmente ou ative **detecção automática**.
4. Escolha o **perfil** (Gaussian, Lorentzian, Pseudo‑Voigt) e se as **larguras** serão **livres** ou **fixas**.
5. Clique em **Ajustar** para ver a soma e os componentes. Baixe parâmetros e curvas.
""")

# Sidebar — Entrada e Pre-processamento
with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader("Arquivo (.csv ou .xlsx) — 2 colunas (X,Y)", type=["csv", "xlsx"])
    decimal = st.selectbox("Separador decimal", [".", ","], index=0)
    delimiter = st.text_input("Delimitador (CSV)", value=",")
    sheet = st.text_input("Planilha (XLSX)", value="")
    use_demo = st.checkbox("Usar dados de demonstração", value=False)

    st.header("Pré-processamento")
    orientation_mode = st.radio("Orientação dos picos", ["auto", "up", "down"], index=0, help="Se os picos estão para cima ou para baixo.")
    do_smooth = st.checkbox("Suavizar (Savitzky-Golay)", value=True)
    win = st.number_input("Janela (ímpar)", min_value=3, value=11, step=2)
    poly = st.number_input("Ordem do polinômio", min_value=1, value=3, step=1)

    do_baseline = st.checkbox("Subtrair linha base (ALS)", value=True)
    lam = st.number_input("λ (ALS)", value=1e5, step=1e4, format="%.0f")
    p_als = st.slider("p (ALS)", 0.0, 1.0, 0.01, step=0.01, value=0.01)
    niter = st.number_input("Iterações (ALS)", min_value=1, value=10, step=1)

    st.header("Detecção de picos")
    auto_detect = st.checkbox("Detectar picos automaticamente", value=False)
    centers_text = st.text_input("Centros manuais (separados por vírgula)", value="")
    prominence_rel = st.number_input("Proeminência mínima (relativa)", value=0.05, step=0.01, min_value=0.0)
    distance_pts = st.number_input("Distância mínima entre picos (pontos)", value=10, step=1, min_value=1)

    st.header("Modelo de ajuste")
    profile = st.selectbox("Perfil", ["Gaussian", "Lorentzian", "Pseudo-Voigt"], index=1)
    fwhm_guess = st.number_input("FWHM inicial (mesma unidade de X)", value=15.0, step=0.5, min_value=0.001)
    widths_free = st.checkbox("Larguras livres (por pico)", value=True)
    allow_shift = st.checkbox("Permitir desvio dos centros", value=True)
    max_shift = st.number_input("Desvio máx. do centro", value=8.0, step=0.5, min_value=0.0)
    nonneg = st.checkbox("Amplitude não-negativa", value=True)

    if profile == "Pseudo-Voigt":
        eta_fixed = st.checkbox("Fixar η (mistura)", value=False)
        eta_value = st.slider("η", 0.0, 1.0, 0.5, 0.01)
    else:
        eta_fixed, eta_value = True, 0.0

# ====== Carregar dados ======
df = None
if use_demo:
    # Gerar sinal sintético (mistura de lorentzianas) com baseline e ruído
    rng = np.random.default_rng(0)
    x = np.linspace(800, 1800, 2000)
    baseline_true = 0.0003*(x-1100) + 0.2*np.sin(x/50.0)
    def L(x, A, x0, g): return A * (g**2)/((x-x0)**2 + g**2)
    y = baseline_true + L(x, 1.5, 1000, 5) + L(x, 1.2, 1248, 8) + L(x, 0.9, 1584, 10)
    y += rng.normal(0, 0.05, size=x.size)
    df = pd.DataFrame({"X": x, "Y": y})
elif uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded, sep=delimiter, decimal=decimal)
        else:
            df = pd.read_excel(uploaded, sheet_name=sheet if sheet else 0)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

if df is not None and df.shape[1] >= 2:
    st.success("Dados carregados.")
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    with c1:
        col_x = st.selectbox("Coluna X", cols, index=0)
    with c2:
        col_y = st.selectbox("Coluna Y", cols, index=1)

    x = pd.to_numeric(df[col_x], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[col_y], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    order = np.argsort(x)
    x = x[order]; y = y[order]

    # ROI
    st.subheader("Faixa de ajuste (ROI)")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    roi = st.slider("Selecione a faixa de X para ajuste", min_value=xmin, max_value=xmax, value=(xmin, xmax))
    roi_mask = (x >= roi[0]) & (x <= roi[1])
    x = x[roi_mask]; y = y[roi_mask]

    # Smoothing
    y_proc = y.copy()
    if do_smooth:
        try:
            win_eff = int(win)
            if win_eff % 2 == 0: win_eff += 1
            win_eff = max(5, min(win_eff, max(5, (len(y_proc)//2)*2-1)))
            poly_eff = int(min(poly, 5))
            y_proc = savgol_filter(y_proc, window_length=win_eff, polyorder=poly_eff, mode="interp")
        except Exception as e:
            st.warning(f"Falha na suavização: {e}")

    # Baseline
    baseline = np.zeros_like(y_proc)
    if do_baseline:
        try:
            baseline = als_baseline(y_proc, lam=float(lam), p=float(p_als), niter=int(niter))
            y_proc = y_proc - baseline
        except Exception as e:
            st.warning(f"Falha na linha base: {e}")

    # Orientação
    if orientation_mode == "auto":
        orient = detect_orientation(y_proc)
    else:
        orient = orientation_mode
    invert_peaks = (orient == "down")

    # Centros
    centers = None
    if auto_detect:
        yp = -y_proc if invert_peaks else y_proc
        prom = max(prominence_rel * max(1e-12, np.ptp(yp)), 1e-12)
        peaks, _ = find_peaks(yp, prominence=prom, distance=int(distance_pts))
        centers = x[peaks]
    else:
        txt = centers_text.strip().replace(";", ",")
        if txt:
            try:
                centers = np.array([float(v) for v in txt.split(",") if v.strip()], dtype=float)
            except Exception as e:
                st.error(f"Não foi possível interpretar os centros: {e}")

    # ---- Plots iniciais
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original"))
    if do_baseline:
        fig0.add_trace(go.Scatter(x=x, y=baseline, mode="lines", name="Linha base"))
        fig0.add_trace(go.Scatter(x=x, y=y - baseline, mode="lines", name="Sem linha base", line=dict(dash="dash")))
    if centers is not None and len(centers) > 0:
        for c in centers:
            fig0.add_vline(x=c, line=dict(width=1, dash="dot"), annotation_text=f"{c:.2f}", annotation_position="top")
    st.plotly_chart(fig0, use_container_width=True)

    # ---- Ajuste
    if centers is not None and len(centers) > 0:
        if st.button("Ajustar agora", type="primary"):
            try:
                p_opt, y_fit, res = fit_model(
                    x, y_proc, centers, profile, fwhm_guess, allow_shift, max_shift,
                    nonneg, widths_free, invert_peaks, eta_fixed, eta_value
                )
                # Componentes
                global_width = None if widths_free else fwhm_guess
                table = params_table(x, p_opt, profile, centers, widths_free, global_width, eta_fixed, eta_value)
                curves = component_curves(x, p_opt, profile, centers, widths_free, global_width, eta_fixed, eta_value)

                r2, rmse = r2_rmse(y_proc, y_fit)

                # Plot do ajuste
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y_proc, mode="lines", name="Processado"))
                fig.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Ajuste (soma)"))
                for label, yi in curves:
                    fig.add_trace(go.Scatter(x=x, y=yi, mode="lines", name=label))
                st.plotly_chart(fig, use_container_width=True)

                # Métricas e tabela
                cA, cB = st.columns([1,1])
                with cA:
                    st.metric("R²", f"{r2:.5f}")
                with cB:
                    st.metric("RMSE", f"{rmse:.6g}")

                st.subheader("Parâmetros por pico")
                st.dataframe(table, use_container_width=True)

                # Downloads
                out_all = pd.DataFrame({"x": x, "y_proc": y_proc, "y_fit": y_fit})
                for k, (label, yi) in enumerate(curves, start=1):
                    out_all[f"comp_{k}"] = yi
                st.download_button("Baixar curvas (CSV)", out_all.to_csv(index=False).encode("utf-8"),
                                   file_name="curvas_deconvolucao.csv", mime="text/csv")
                st.download_button("Baixar parâmetros (CSV)", table.to_csv(index=False).encode("utf-8"),
                                   file_name="parametros_deconvolucao.csv", mime="text/csv")
                settings = dict(
                    profile=profile, centers=list(map(float, centers)), fwhm_guess=float(fwhm_guess),
                    widths_free=bool(widths_free), allow_shift=bool(allow_shift), max_shift=float(max_shift),
                    nonneg=bool(nonneg), invert_peaks=bool(invert_peaks), orientation=orient,
                    smooth=dict(enabled=bool(do_smooth), window=int(win), poly=int(poly)),
                    baseline=dict(enabled=bool(do_baseline), lam=float(lam), p=float(p_als), niter=int(niter)),
                    detection=dict(auto=bool(auto_detect), prominence_rel=float(prominence_rel), distance_pts=int(distance_pts)),
                    roi=dict(xmin=float(roi[0]), xmax=float(roi[1])),
                    eta=dict(fixed=bool(eta_fixed), value=float(eta_value))
                )
                st.download_button("Baixar configurações (JSON)",
                                   data=json.dumps(settings, indent=2).encode("utf-8"),
                                   file_name="config_deconvolucao.json",
                                   mime="application/json")
                st.success("Ajuste concluído.")
            except Exception as e:
                st.error(f"Falha no ajuste: {e}")
    else:
        st.info("Informe centros manualmente ou ative a detecção automática para habilitar o ajuste.")
else:
    st.info("Carregue um arquivo ou ative 'Usar dados de demonstração' na barra lateral.")


