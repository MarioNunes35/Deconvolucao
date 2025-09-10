#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deconvolução Raman (1500–1700 cm⁻¹) — Streamlit
• Bandas ajustáveis (Centro, Amplitude, Largura/FWHM, Forma Gaussiana↔Lorentziana)
• Leitura de CSV/Excel (detecta separador), ordenação e filtro por faixa espectral
• Gráficos interativos com Plotly (Espectro, Ajuste, Bandas, Resíduos)
• Exportação JSON (parâmetros + estatísticas + espectro/ajuste/resíduos)
Notas de robustez em relação ao original:
- Removido @st.cache_data em funções puras (numpy arrays podem causar hashing em algumas versões)
- Fallback para add_hline ausente (algumas versões antigas do plotly) usando shapes
- Leitura de CSV com detecção automática de separador (sep=None, engine='python')
- Mensagens de erro mais claras e validações adicionais
- Chaves únicas e estáveis para sliders (c_i, a_i, w_i, s_i)
- Exemplo integrado (botão) para testes rápidos
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Config da página (precisa vir antes de outros st.*) ----------
try:
    st.set_page_config(page_title="Deconvolução Raman", page_icon="🔬", layout="wide")
except Exception:
    # Em alguns ambientes (re-run / multipages) set_page_config pode já ter sido chamado
    pass


# ---------- Funções matemáticas ----------
def gaussian(x, center, amplitude, width):
    """Gaussiana (FWHM -> sigma)."""
    sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, amplitude, width):
    """Lorentziana (FWHM -> gamma)."""
    gamma = width / 2.0
    return amplitude * (gamma**2) / ((x - center) ** 2 + gamma**2)


def mixed_function(x, center, amplitude, width, shape_factor):
    """
    Mistura Gaussiana-Lorentziana (shape_factor 0.0=Gauss, 1.0=Lorenz).
    """
    if shape_factor <= 0.0:
        return gaussian(x, center, amplitude, width)
    if shape_factor >= 1.0:
        return lorentzian(x, center, amplitude, width)
    g = gaussian(x, center, amplitude, width)
    l = lorentzian(x, center, amplitude, width)
    return (1.0 - shape_factor) * g + shape_factor * l


def calculate_area(amplitude, width, shape_factor):
    """
    Área aproximada sob a banda (analítica para G e L, combinação linear para mistura).
    """
    # Área Gaussiana usando FWHM
    gauss_area = amplitude * width * np.sqrt(2.0 * np.pi) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # Área Lorentziana usando FWHM
    lorenz_area = amplitude * width * np.pi / 2.0
    if shape_factor <= 0.0:
        return gauss_area
    if shape_factor >= 1.0:
        return lorenz_area
    return (1.0 - shape_factor) * gauss_area + shape_factor * lorenz_area


# ---------- Dados de exemplo ----------
def load_example_data():
    wavenumbers = np.arange(1500, 1701, 1.0)
    baseline = 2000.0
    intensity = np.full_like(wavenumbers, baseline, dtype=float)

    # (center, amplitude, width(FWHM), shape_factor [0=G .. 1=L])
    example_bands = [
        (1547.0, 8000.0, 15.0, 0.3),
        (1566.0, 6000.0, 12.0, 0.1),
        (1580.0, 12000.0, 18.0, 0.0),
        (1604.0, 10000.0, 16.0, 0.2),
        (1620.0, 15000.0, 20.0, 0.4),
        (1632.0, 7000.0, 13.0, 0.0),
        (1667.0, 5000.0, 22.0, 0.6),
    ]

    for c, a, w, s in example_bands:
        intensity += mixed_function(wavenumbers, c, a, w, s)

    df = pd.DataFrame({"wavenumber": wavenumbers, "intensity": intensity})
    return df


# ---------- Leitura e preparo de arquivo ----------
def process_file(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            # autodetecta separador e decimal
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            df = pd.read_excel(uploaded_file)

        # Normaliza nomes de colunas e pega as 2 primeiras como wavenumber/intensity
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("Arquivo precisa conter ao menos 2 colunas (wavenumber, intensity).")

        df = df.rename(columns={cols[0]: "wavenumber", cols[1]: "intensity"})
        # Converte para numérico e trata não-numéricos
        df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df = df.dropna(subset=["wavenumber", "intensity"])

        # Filtra faixa 1500–1700 cm⁻¹
        df = df[(df["wavenumber"] >= 1500.0) & (df["wavenumber"] <= 1700.0)]
        df = df.sort_values("wavenumber").reset_index(drop=True)
        if df.empty:
            raise ValueError("Após o filtro 1500–1700 cm⁻¹, não restaram pontos válidos.")

        return df
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
        return None


# ---------- UI principal ----------
def main():
    st.title("🔬 Deconvolução Espectral Raman")
    st.markdown("**Modelos: Gaussiana + Lorentziana (mistura)** · **Faixa: 1500–1700 cm⁻¹**")

    # Sidebar — dados e configurações gerais
    with st.sidebar:
        st.header("📁 Dados")
        uploaded_file = st.file_uploader("Arquivo Excel/CSV", type=["xlsx", "xls", "csv"])
        use_example = st.button("🧪 Carregar dados de exemplo")

        st.divider()
        st.header("⚙️ Configurações")
        baseline = st.slider("Baseline (offset)", 0, 20000, 2000, 100)
        show_bands = st.checkbox("Mostrar bandas individuais", True)
        show_residuals = st.checkbox("Mostrar resíduos (exp - ajuste)", True)

    # Carregamento dos dados
    if uploaded_file is not None:
        data = process_file(uploaded_file)
        if data is None:
            return
        st.success(f"✅ {len(data)} pontos carregados")
    elif use_example:
        data = load_example_data()
        st.success("✅ Dados de exemplo carregados")
    else:
        st.info("👆 Carregue um arquivo ou use os dados de exemplo para continuar.")
        return

    # Estado inicial de bandas
    if "bands" not in st.session_state:
        st.session_state.bands = [
            {"name": "Banda 1", "center": 1547.0, "amplitude": 5000, "width": 15.0, "shape": 0.0, "color": "#ff6b6b"},
            {"name": "Banda 2", "center": 1566.0, "amplitude": 4000, "width": 12.0, "shape": 0.0, "color": "#4ecdc4"},
            {"name": "Banda 3", "center": 1580.0, "amplitude": 8000, "width": 18.0, "shape": 0.0, "color": "#45b7d1"},
            {"name": "Banda 4", "center": 1604.0, "amplitude": 6000, "width": 16.0, "shape": 0.0, "color": "#96ceb4"},
            {"name": "Banda 5", "center": 1620.0, "amplitude": 5000, "width": 14.0, "shape": 0.0, "color": "#ffeaa7"},
        ]

    max_amp = max(50000, int(data["intensity"].max() * 1.5))

    # Layout principal
    col_controls, col_plot = st.columns([1, 2])

    with col_controls:
        st.header("🎛️ Controles")
        for i, band in enumerate(st.session_state.bands):
            with st.expander(f"{band['name']}", expanded=i < 3):
                center = st.slider("Centro (cm⁻¹)", 1500.0, 1700.0, float(band["center"]), 0.5, key=f"c_{i}")
                amplitude = st.slider("Amplitude", 0, max_amp, int(band["amplitude"]), 100, key=f"a_{i}")
                width = st.slider("FWHM (cm⁻¹)", 3.0, 80.0, float(band["width"]), 0.5, key=f"w_{i}")
                shape = st.slider("Forma (0=Gauss, 1=Lorenz)", 0.0, 1.0, float(band["shape"]), 0.05, key=f"s_{i}")

                st.session_state.bands[i].update(
                    {"center": center, "amplitude": amplitude, "width": width, "shape": shape}
                )

                if shape == 0.0:
                    st.caption("Modelo: **Gaussiana**")
                elif shape == 1.0:
                    st.caption("Modelo: **Lorentziana**")
                else:
                    st.caption(f"Modelo: **Misto** (G:{1.0-shape:.2f} · L:{shape:.2f})")

    with col_plot:
        st.header("📊 Gráfico")

        # Vetores
        x = data["wavenumber"].to_numpy(dtype=float)
        exp = data["intensity"].to_numpy(dtype=float)

        # Ajuste total
        fitted = np.full_like(x, float(baseline), dtype=float)
        contributions = []
        for band in st.session_state.bands:
            contrib = mixed_function(x, band["center"], band["amplitude"], band["width"], band["shape"])
            fitted += contrib
            contributions.append(contrib)

        # Estatísticas básicas
        residuals = exp - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((exp - np.mean(exp))**2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        # Gráfico principal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=exp, mode="lines", name="Experimental", line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=x, y=fitted, mode="lines", name="Ajuste", line=dict(color="red", width=2, dash="dash")))
        fig.add_trace(
            go.Scatter(x=x, y=np.full_like(x, baseline), mode="lines", name="Baseline", line=dict(color="gray", width=1, dash="dot"))
        )

        if show_bands:
            for band, contrib in zip(st.session_state.bands, contributions):
                fig.add_trace(
                    go.Scatter(x=x, y=baseline + contrib, mode="lines", name=band["name"], line=dict(color=band["color"], width=2), opacity=0.7)
                )

        fig.update_layout(title="Deconvolução", xaxis_title="Raman Shift (cm⁻¹)", yaxis_title="Intensidade", height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Resíduos
        if show_residuals:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=x, y=residuals, mode="lines+markers", name="Resíduos"))
            # Fallback para add_hline ausente
            try:
                fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
            except Exception:
                fig_res.update_layout(
                    shapes=[dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0, line=dict(dash="dash", width=1))]
                )

            fig_res.update_layout(title="Resíduos (Experimental - Ajuste)", xaxis_title="Raman Shift (cm⁻¹)", yaxis_title="Intensidade", height=300)
            st.plotly_chart(fig_res, use_container_width=True)

    # Métricas
    st.header("📈 Resultados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.4f}")
    c2.metric("RMSE", f"{rmse:.1f}")
    c3.metric("MAE", f"{mae:.1f}")
    c4.metric("Pontos", len(data))

    # Tabela de parâmetros
    st.subheader("📋 Parâmetros das Bandas")
    rows = []
    for band in st.session_state.bands:
        area = calculate_area(band["amplitude"], band["width"], band["shape"])
        curve_type = "Gaussiana" if band["shape"] == 0.0 else ("Lorentziana" if band["shape"] == 1.0 else "Mista")
        rows.append(
            {
                "Banda": band["name"],
                "Centro (cm⁻¹)": f"{band['center']:.1f}",
                "Amplitude": int(band["amplitude"]),
                "FWHM (cm⁻¹)": f"{band['width']:.1f}",
                "Tipo": curve_type,
                "Área (aprox.)": f"{area:,.0f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Exportação JSON
    st.subheader("💾 Exportar")
    if st.button("📊 Baixar Resultados (JSON)"):
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": {"r2": r2, "rmse": rmse, "mae": mae},
            "baseline": baseline,
            "bands": [
                {
                    "name": b["name"],
                    "center": float(b["center"]),
                    "amplitude": int(b["amplitude"]),
                    "width": float(b["width"]),
                    "shape_factor": float(b["shape"]),
                    "area": float(calculate_area(b["amplitude"], b["width"], b["shape"])),
                }
                for b in st.session_state.bands
            ],
            "spectrum": pd.DataFrame({"wavenumber": x, "experimental": exp, "fitted": fitted, "residual": residuals}).to_dict(
                orient="records"
            ),
        }
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download JSON",
            json_str,
            file_name=f"raman_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()


