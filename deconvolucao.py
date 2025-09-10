
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares
import plotly.graph_objects as go

st.set_page_config(page_title="Deconvolução Raman (Gaussian/Lorentzian)", layout="wide")

# ------------------------ Helpers ------------------------
def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline (Eilers & Boelens, 2005)
    lam: suavização; p: assimetria (0<p<1); niter: iterações.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sp.diags(w, 0, shape=(L, L))
        Z = W + lam * D @ D.T
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return np.asarray(z)

def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def lorentzian(x, A, x0, gamma):
    return A * (gamma**2) / ((x - x0)**2 + gamma**2)

def pseudo_voigt(x, A, x0, sigma, gamma, eta):
    # eta in [0,1], mix of gaussian (1-eta) and lorentzian eta
    return (1-eta) * gaussian(x, A, x0, sigma) + eta * lorentzian(x, A, x0, gamma)

def fwhm_gaussian(sigma):
    return 2.354820045 * sigma

def sigma_from_fwhm(fwhm):
    return fwhm / 2.354820045

def gamma_from_fwhm(fwhm):
    return fwhm / 2.0

def area_gaussian(A, sigma):
    return A * sigma * np.sqrt(2*np.pi)

def area_lorentzian(A, gamma):
    return A * np.pi * gamma

def initial_guess_from_centers(x, y, centers, profile, fwhm_guess, invert_peaks):
    y2 = -y if invert_peaks else y
    A_guess = []
    w_guess = []
    for c in centers:
        idx = (np.abs(x - c)).argmin()
        A_guess.append(max(y2[idx] - np.median(y2), 1e-6))
        w_guess.append(fwhm_guess)
    A_guess = np.array(A_guess)
    w_guess = np.array(w_guess)

    params = []
    for A, c, w in zip(A_guess, centers, w_guess):
        if profile == "Gaussian":
            sigma = sigma_from_fwhm(w)
            params.extend([A, c, sigma])
        elif profile == "Lorentzian":
            gamma = gamma_from_fwhm(w)
            params.extend([A, c, gamma])
        elif profile == "Pseudo-Voigt":
            sigma = sigma_from_fwhm(w)
            gamma = gamma_from_fwhm(w)
            params.extend([A, c, sigma, gamma, 0.5])
    return np.array(params, dtype=float)

def model_sum(x, params, profile):
    y = np.zeros_like(x, dtype=float)
    i = 0
    if profile == "Gaussian":
        while i + 2 < len(params):
            A, x0, sigma = params[i:i+3]
            y += gaussian(x, A, x0, sigma)
            i += 3
    elif profile == "Lorentzian":
        while i + 2 < len(params):
            A, x0, gamma = params[i:i+3]
            y += lorentzian(x, A, x0, gamma)
            i += 3
    elif profile == "Pseudo-Voigt":
        while i + 4 < len(params):
            A, x0, sigma, gamma, eta = params[i:i+5]
            eta = np.clip(eta, 0.0, 1.0)
            y += pseudo_voigt(x, A, x0, sigma, gamma, eta)
            i += 5
    return y

def fit_peaks(x, y, centers, profile, fwhm_guess, allow_shift, max_shift, nonneg, invert_peaks):
    y_fit_target = -y if invert_peaks else y
    p0 = initial_guess_from_centers(x, y_fit_target, centers, profile, fwhm_guess, invert_peaks)

    # bounds
    lb = []
    ub = []
    for c in centers:
        if profile == "Gaussian":
            lb += [0.0 if nonneg else -np.inf, c - max_shift if allow_shift else c, 1e-6]
            ub += [np.inf, c + max_shift if allow_shift else c, np.inf]
        elif profile == "Lorentzian":
            lb += [0.0 if nonneg else -np.inf, c - max_shift if allow_shift else c, 1e-6]
            ub += [np.inf, c + max_shift if allow_shift else c, np.inf]
        elif profile == "Pseudo-Voigt":
            lb += [0.0 if nonneg else -np.inf, c - max_shift if allow_shift else c, 1e-6, 1e-6, 0.0]
            ub += [np.inf, c + max_shift if allow_shift else c, np.inf,  np.inf, 1.0]

    def residuals(p):
        return model_sum(x, p, profile) - y_fit_target

    res = least_squares(residuals, p0, bounds=(np.array(lb), np.array(ub)))
    p_opt = res.x
    return p_opt, res.cost, res.success

def params_to_table(params, profile):
    rows = []
    i = 0
    idx = 1
    if profile == "Gaussian":
        while i + 2 < len(params):
            A, x0, sigma = params[i:i+3]
            fwhm = fwhm_gaussian(sigma)
            area = area_gaussian(A, sigma)
            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm=fwhm, area=area))
            idx += 1
            i += 3
    elif profile == "Lorentzian":
        while i + 2 < len(params):
            A, x0, gamma = params[i:i+3]
            fwhm = 2*gamma
            area = area_lorentzian(A, gamma)
            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm=fwhm, area=area))
            idx += 1
            i += 3
    elif profile == "Pseudo-Voigt":
        while i + 4 < len(params):
            A, x0, sigma, gamma, eta = params[i:i+5]
            fwhm_g = fwhm_gaussian(sigma)
            fwhm_l = 2*gamma
            area = (1-eta)*area_gaussian(A, sigma) + eta*area_lorentzian(A, gamma)
            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm_gaussian=fwhm_g, fwhm_lorentzian=fwhm_l, eta=eta, area_est=area))
            idx += 1
            i += 5
    return pd.DataFrame(rows)

# ------------------------ UI ------------------------
st.title("🔬 Deconvolução Espectral Raman — Streamlit")

with st.sidebar:
    st.header("Configurações & Entrada")

    uploaded = st.file_uploader("Arquivo (.csv ou .xlsx) com duas colunas (x, y)", type=["csv", "xlsx"])
    delimiter = st.text_input("Delimitador (CSV)", value=",")
    sheet = st.text_input("Planilha (XLSX)", value="")

    invert_peaks = st.checkbox("Picos para baixo (inverter y)", value=False)

    do_smooth = st.checkbox("Suavizar (Savitzky-Golay)", value=True)
    win = st.number_input("Janela (ímpar)", min_value=3, value=11, step=2)
    poly = st.number_input("Ordem do polinômio", min_value=1, value=3, step=1)

    do_baseline = st.checkbox("Subtrair linha base (ALS)", value=True)
    lam = st.number_input("λ (ALS)", value=1e5, step=1e4, format="%.0f")
    p_als = st.slider("p (ALS)", 0.0, 1.0, 0.01, step=0.01)
    niter = st.number_input("Iterações (ALS)", min_value=1, value=10, step=1)

    st.divider()
    st.subheader("Picos & Ajuste")
    profile = st.selectbox("Perfil dos picos", ["Gaussian", "Lorentzian", "Pseudo-Voigt"], index=1)
    centers_text = st.text_input("Centros (cm⁻¹) separados por vírgula", value="")
    auto_detect = st.checkbox("Detectar picos automaticamente (ignora centros acima)", value=False)
    prominence = st.number_input("Proeminência mínima p/ detecção (relativa)", value=0.05, step=0.01)

    fwhm_guess = st.number_input("FWHM inicial (cm⁻¹)", value=15.0, step=1.0)
    allow_shift = st.checkbox("Permitir desvio dos centros", value=True)
    max_shift = st.number_input("Desvio máx. (cm⁻¹)", value=8.0, step=0.5)
    nonneg = st.checkbox("Amplitude não-negativa", value=True)

    st.divider()
    st.caption("Dica: Você pode clicar no ícone da câmera no gráfico para baixar PNG sem precisar do Kaleido.")

df = None
x = y = None

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded, sep=delimiter)
        else:
            df = pd.read_excel(uploaded, sheet_name=sheet if sheet else 0)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

if df is not None:
    st.success(f"Arquivo carregado: {uploaded.name}")
    cols = list(df.columns)
    col_x = st.selectbox("Coluna X", cols, index=0 if len(cols)>0 else None)
    col_y = st.selectbox("Coluna Y", cols, index=1 if len(cols)>1 else None)

    x = np.asarray(pd.to_numeric(df[col_x], errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(df[col_y], errors="coerce"), dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_proc = y.copy()
    if do_smooth:
        try:
            if win % 2 == 0:
                win += 1
            # janela não pode exceder len(y_proc)-1 e deve ser ímpar
            win_eff = min(win, max(5, (len(y_proc)//2)*2-1))
            win_eff = win_eff if win_eff % 2 == 1 else win_eff-1
            y_proc = savgol_filter(y_proc, window_length=max(5, win_eff), polyorder=min(poly, 5))
        except Exception as e:
            st.warning(f"Falha na suavização: {e}")

    baseline = np.zeros_like(y_proc)
    if do_baseline:
        try:
            baseline = als_baseline(y_proc, lam=lam, p=p_als, niter=int(niter))
            y_proc = y_proc - baseline
        except Exception as e:
            st.warning(f"Falha na linha base: {e}")

    centers = None
    if auto_detect:
        yp = -y_proc if invert_peaks else y_proc
        if np.ptp(yp) > 0:
            prom = prominence * np.ptp(yp)
        else:
            prom = prominence
        peaks, _ = find_peaks(yp, prominence=max(prom, 1e-9))
        centers = x[peaks]
        centers = np.round(centers, 4)
    else:
        if centers_text.strip():
            try:
                centers = np.array([float(v) for v in centers_text.replace(";", ",").split(",") if v.strip()], dtype=float)
            except Exception as e:
                st.error(f"Não foi possível interpretar os centros: {e}")
        else:
            st.info("Informe os centros ou ative a detecção automática.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original"))
    if do_baseline:
        fig.add_trace(go.Scatter(x=x, y=baseline, mode="lines", name="Linha base"))
        fig.add_trace(go.Scatter(x=x, y=y - baseline, mode="lines", name="Sem linha base", line=dict(dash="dash")))
    st.plotly_chart(fig, use_container_width=True)

    if centers is not None and len(centers) > 0:
        if st.button("Ajustar picos agora", type="primary"):
            try:
                params, cost, ok = fit_peaks(x, y_proc, centers, profile, fwhm_guess, allow_shift, max_shift, nonneg, invert_peaks)
                y_fit = model_sum(x, params, profile)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x, y=y_proc, mode="lines", name="Processado"))
                fig2.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Ajuste soma"))
                i = 0
                if profile == "Gaussian":
                    while i + 2 < len(params):
                        A, x0, sigma = params[i:i+3]
                        yi = gaussian(x, A, x0, sigma)
                        fig2.add_trace(go.Scatter(x=x, y=yi, mode="lines", name=f"Pico @ {x0:.2f}"))
                        i += 3
                elif profile == "Lorentzian":
                    while i + 2 < len(params):
                        A, x0, gamma = params[i:i+3]
                        yi = lorentzian(x, A, x0, gamma)
                        fig2.add_trace(go.Scatter(x=x, y=yi, mode="lines", name=f"Pico @ {x0:.2f}"))
                        i += 3
                elif profile == "Pseudo-Voigt":
                    while i + 4 < len(params):
                        A, x0, sigma, gamma, eta = params[i:i+5]
                        yi = pseudo_voigt(x, A, x0, sigma, gamma, eta)
                        fig2.add_trace(go.Scatter(x=x, y=yi, mode="lines", name=f"Pico @ {x0:.2f} (η={eta:.2f})"))
                        i += 5
                st.plotly_chart(fig2, use_container_width=True)

                # tabela
                def params_to_table(params, profile):
                    rows = []
                    i = 0
                    idx = 1
                    if profile == "Gaussian":
                        while i + 2 < len(params):
                            A, x0, sigma = params[i:i+3]
                            fwhm = 2.354820045 * sigma
                            area = A * sigma * np.sqrt(2*np.pi)
                            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm=fwhm, area=area))
                            idx += 1
                            i += 3
                    elif profile == "Lorentzian":
                        while i + 2 < len(params):
                            A, x0, gamma = params[i:i+3]
                            fwhm = 2*gamma
                            area = np.pi * gamma * A
                            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm=fwhm, area=area))
                            idx += 1
                            i += 3
                    elif profile == "Pseudo-Voigt":
                        while i + 4 < len(params):
                            A, x0, sigma, gamma, eta = params[i:i+5]
                            fwhm_g = 2.354820045 * sigma
                            fwhm_l = 2*gamma
                            area = (1-eta)*(A*sigma*np.sqrt(2*np.pi)) + eta*(np.pi*gamma*A)
                            rows.append(dict(peak=idx, center=x0, amplitude=A, fwhm_gaussian=fwhm_g, fwhm_lorentzian=fwhm_l, eta=eta, area_est=area))
                            idx += 1
                            i += 5
                    return pd.DataFrame(rows)

                table = params_to_table(params, profile)
                st.subheader("Parâmetros ajustados")
                st.dataframe(table, use_container_width=True)

                st.download_button("Baixar parâmetros (CSV)",
                                   data=table.to_csv(index=False).encode("utf-8"),
                                   file_name="parametros_ajuste.csv",
                                   mime="text/csv")

                out = pd.DataFrame({"x": x, "y_processado": y_proc, "y_fit": y_fit})
                st.download_button("Baixar curvas (CSV)",
                                   data=out.to_csv(index=False).encode("utf-8"),
                                   file_name="curvas_ajuste.csv",
                                   mime="text/csv")

                settings = dict(profile=profile,
                                centers=list(map(float, centers)),
                                fwhm_guess=float(fwhm_guess),
                                allow_shift=bool(allow_shift),
                                max_shift=float(max_shift),
                                nonneg=bool(nonneg),
                                invert_peaks=bool(invert_peaks),
                                smooth=dict(enabled=bool(do_smooth), window=int(win), poly=int(poly)),
                                baseline=dict(enabled=bool(do_baseline), lam=float(lam), p=float(p_als), niter=int(niter)))
                st.download_button("Baixar configurações (JSON)",
                                   data=json.dumps(settings, indent=2).encode("utf-8"),
                                   file_name="config_ajuste.json",
                                   mime="application/json")

                st.success("Ajuste concluído." if ok else "Ajuste finalizado (resíduo não nulo).")
            except Exception as e:
                st.error(f"Falha no ajuste: {e}")
    else:
        st.info("Defina os centros ou ative a detecção automática para habilitar o ajuste.")

else:
    st.info("Faça upload de um arquivo para começar. Você também pode testar com o exemplo abaixo.")
    try:
        with open('exemplo_raman.csv', 'rb') as f:
            st.download_button("Baixar exemplo_raman.csv", f, file_name="exemplo_raman.csv", mime="text/csv")
    except Exception:
        pass


