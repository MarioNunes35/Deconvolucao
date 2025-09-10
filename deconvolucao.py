#!/usr/bin/env python3
"""
Aplicação de Deconvolução Raman - Versão Streamlit Simplificada
Suporte a Gaussiana e Lorentziana
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Configuração da página
try:
    st.set_page_config(
        page_title="Deconvolução Raman",
        page_icon="🔬",
        layout="wide"
    )
except:
    pass

@st.cache_data
def gaussian(x, center, amplitude, width):
    """Função Gaussiana"""
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

@st.cache_data
def lorentzian(x, center, amplitude, width):
    """Função Lorentziana"""
    gamma = width / 2
    return amplitude * (gamma**2) / ((x - center)**2 + gamma**2)

def mixed_function(x, center, amplitude, width, shape_factor):
    """Função mista Gaussiana-Lorentziana"""
    if shape_factor == 0:
        return gaussian(x, center, amplitude, width)
    elif shape_factor == 1:
        return lorentzian(x, center, amplitude, width)
    else:
        gauss = gaussian(x, center, amplitude, width)
        lorenz = lorentzian(x, center, amplitude, width)
        return (1 - shape_factor) * gauss + shape_factor * lorenz

def calculate_area(amplitude, width, shape_factor):
    """Calcula área aproximada"""
    if shape_factor == 0:  # Gaussiana
        return amplitude * width * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.log(2)))
    elif shape_factor == 1:  # Lorentziana
        return amplitude * width * np.pi / 2
    else:  # Mista
        gauss_area = amplitude * width * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.log(2)))
        lorenz_area = amplitude * width * np.pi / 2
        return (1 - shape_factor) * gauss_area + shape_factor * lorenz_area

@st.cache_data
def load_example_data():
    """Gera dados de exemplo"""
    wavenumbers = np.arange(1500, 1701, 1)
    baseline = 2000
    intensity = np.full_like(wavenumbers, baseline, dtype=float)
    
    # Bandas de exemplo
    example_bands = [
        (1547, 8000, 15, 0.3),
        (1566, 6000, 12, 0.1),
        (1580, 12000, 18, 0.0),
        (1604, 10000, 16, 0.2),
        (1620, 15000, 20, 0.4),
        (1632, 7000, 13, 0.0),
        (1667, 5000, 22, 0.6)
    ]
    
    for center, amplitude, width, shape in example_bands:
        intensity += mixed_function(wavenumbers, center, amplitude, width, shape)
    
    # Ruído
    noise = np.random.normal(0, 300, len(wavenumbers))
    intensity = np.maximum(intensity + noise, 0)
    
    return pd.DataFrame({
        'wavenumber': wavenumbers,
        'intensity': intensity
    })

def process_file(uploaded_file):
    """Processa arquivo carregado"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Assumir primeiras duas colunas
        df.columns = ['wavenumber', 'intensity'] + list(df.columns[2:])
        
        # Filtrar região
        df = df[(df['wavenumber'] >= 1500) & (df['wavenumber'] <= 1700)]
        df = df.dropna()
        df = df.sort_values('wavenumber').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return None

def main():
    st.title("🔬 Deconvolução Espectral Raman")
    st.markdown("**Gaussiana + Lorentziana | Região 1500-1700 cm⁻¹**")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Dados")
        
        uploaded_file = st.file_uploader(
            "Arquivo Excel/CSV",
            type=['xlsx', 'xls', 'csv']
        )
        
        use_example = st.button("🧪 Dados de Exemplo")
        
        st.divider()
        
        st.header("⚙️ Configurações")
        baseline = st.slider("Baseline", 0, 20000, 2000, 100)
        show_bands = st.checkbox("Mostrar bandas", True)
        show_residuals = st.checkbox("Mostrar resíduos", True)
    
    # Carregar dados
    if uploaded_file:
        data = process_file(uploaded_file)
        if data is None:
            return
        st.success(f"✅ {len(data)} pontos carregados")
    elif use_example:
        data = load_example_data()
        st.success("✅ Dados de exemplo carregados")
    else:
        st.info("👆 Carregue dados ou use exemplo")
        return
    
    # Inicializar bandas
    if 'bands' not in st.session_state:
        st.session_state.bands = [
            {'name': 'Banda 1', 'center': 1547, 'amplitude': 5000, 'width': 15, 'shape': 0.0, 'color': '#ff6b6b'},
            {'name': 'Banda 2', 'center': 1566, 'amplitude': 4000, 'width': 12, 'shape': 0.0, 'color': '#4ecdc4'},
            {'name': 'Banda 3', 'center': 1580, 'amplitude': 8000, 'width': 18, 'shape': 0.0, 'color': '#45b7d1'},
            {'name': 'Banda 4', 'center': 1604, 'amplitude': 6000, 'width': 16, 'shape': 0.0, 'color': '#96ceb4'},
            {'name': 'Banda 5', 'center': 1620, 'amplitude': 5000, 'width': 14, 'shape': 0.0, 'color': '#ffeaa7'}
        ]
    
    max_amp = max(50000, int(data['intensity'].max() * 1.5))
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎛️ Controles")
        
        for i, band in enumerate(st.session_state.bands):
            with st.expander(f"{band['name']}", expanded=i < 3):
                
                center = st.slider(
                    "Centro", 1500.0, 1700.0, 
                    float(band['center']), 0.5, 
                    key=f"c_{i}"
                )
                
                amplitude = st.slider(
                    "Amplitude", 0, max_amp, 
                    int(band['amplitude']), 100,
                    key=f"a_{i}"
                )
                
                width = st.slider(
                    "FWHM", 3.0, 80.0, 
                    float(band['width']), 0.5,
                    key=f"w_{i}"
                )
                
                shape = st.slider(
                    "Tipo (0=Gauss, 1=Lorenz)", 
                    0.0, 1.0, float(band['shape']), 0.05,
                    key=f"s_{i}"
                )
                
                # Atualizar
                st.session_state.bands[i].update({
                    'center': center,
                    'amplitude': amplitude,
                    'width': width,
                    'shape': shape
                })
                
                # Mostrar tipo
                if shape == 0:
                    st.caption("Gaussiana")
                elif shape == 1:
                    st.caption("Lorentziana")
                else:
                    st.caption(f"Mista (G:{1-shape:.2f}, L:{shape:.2f})")
    
    with col2:
        st.header("📊 Gráfico")
        
        # Calcular ajuste
        x = data['wavenumber'].values
        exp = data['intensity'].values
        
        fitted = np.full_like(x, baseline, dtype=float)
        contributions = []
        
        for band in st.session_state.bands:
            contrib = mixed_function(x, band['center'], band['amplitude'], 
                                   band['width'], band['shape'])
            fitted += contrib
            contributions.append(contrib)
        
        # Estatísticas
        residuals = exp - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((exp - np.mean(exp))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # Gráfico principal
        fig = go.Figure()
        
        # Experimental
        fig.add_trace(go.Scatter(
            x=x, y=exp, mode='lines',
            name='Experimental', line=dict(color='black', width=2)
        ))
        
        # Ajuste
        fig.add_trace(go.Scatter(
            x=x, y=fitted, mode='lines',
            name='Ajuste', line=dict(color='red', width=2, dash='dash')
        ))
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=x, y=np.full_like(x, baseline), mode='lines',
            name='Baseline', line=dict(color='gray', width=1, dash='dot')
        ))
        
        # Bandas individuais
        if show_bands:
            for i, (band, contrib) in enumerate(zip(st.session_state.bands, contributions)):
                fig.add_trace(go.Scatter(
                    x=x, y=baseline + contrib, mode='lines',
                    name=band['name'], line=dict(color=band['color'], width=2),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Deconvolução",
            xaxis_title="Raman Shift (cm⁻¹)",
            yaxis_title="Intensidade",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Resíduos
        if show_residuals:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=x, y=residuals, mode='lines+markers',
                name='Resíduos', line=dict(color='blue', width=1)
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_res.update_layout(
                title="Resíduos",
                xaxis_title="Raman Shift (cm⁻¹)",
                yaxis_title="Resíduo",
                height=250
            )
            st.plotly_chart(fig_res, use_container_width=True)
    
    # Resultados
    st.header("📈 Resultados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.1f}")
    with col3:
        st.metric("MAE", f"{mae:.1f}")
    with col4:
        st.metric("Pontos", len(data))
    
    # Tabela
    st.subheader("📋 Parâmetros")
    
    results = []
    for band in st.session_state.bands:
        area = calculate_area(band['amplitude'], band['width'], band['shape'])
        curve_type = "Gaussiana" if band['shape'] == 0 else \
                    "Lorentziana" if band['shape'] == 1 else "Mista"
        
        results.append({
            'Banda': band['name'],
            'Centro': f"{band['center']:.1f}",
            'Amplitude': f"{band['amplitude']:,}",
            'FWHM': f"{band['width']:.1f}",
            'Tipo': curve_type,
            'Área': f"{area:,.0f}"
        })
    
    st.dataframe(results, use_container_width=True)
    
    # Exportação simplificada
    st.subheader("💾 Exportar")
    
    if st.button("📊 Baixar Resultados (JSON)"):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': {'r2': r2, 'rmse': rmse, 'mae': mae},
            'baseline': baseline,
            'bands': [
                {
                    'name': b['name'],
                    'center': b['center'],
                    'amplitude': b['amplitude'],
                    'width': b['width'],
                    'shape_factor': b['shape'],
                    'area': calculate_area(b['amplitude'], b['width'], b['shape'])
                }
                for b in st.session_state.bands
            ],
            'spectrum': data.assign(fitted=fitted, residual=residuals).to_dict('records')
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"raman_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json"
        )

if __name__ == "__main__":
    main()


