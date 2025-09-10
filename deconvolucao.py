#!/usr/bin/env python3
"""
Aplicação de Deconvolução Raman para Streamlit
Versão com suporte a Gaussiana e Lorentziana
Controles ajustados para altas intensidades
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import base64
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Deconvolução Espectral Raman",
    page_icon="🔬",
    layout="wide"
)

def gaussian(x, center, amplitude, width):
    """Função Gaussiana"""
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian(x, center, amplitude, width):
    """Função Lorentziana"""
    gamma = width / 2
    return amplitude * (gamma**2) / ((x - center)**2 + gamma**2)

def mixed_function(x, center, amplitude, width, shape_factor):
    """Função mista Gaussiana-Lorentziana (Voigt aproximada)
    shape_factor: 0 = pura Gaussiana, 1 = pura Lorentziana"""
    gauss = gaussian(x, center, amplitude, width)
    lorenz = lorentzian(x, center, amplitude, width)
    return (1 - shape_factor) * gauss + shape_factor * lorenz

def calculate_area(amplitude, width, shape_factor):
    """Calcula área aproximada baseada no tipo de curva"""
    if shape_factor == 0:  # Gaussiana
        return amplitude * width * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.log(2)))
    elif shape_factor == 1:  # Lorentziana
        return amplitude * width * np.pi / 2
    else:  # Mista
        gauss_area = amplitude * width * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.log(2)))
        lorenz_area = amplitude * width * np.pi / 2
        return (1 - shape_factor) * gauss_area + shape_factor * lorenz_area

def load_example_data():
    """Gera dados de exemplo simulando espectro Raman"""
    wavenumbers = np.arange(1500, 1701, 1)
    
    # Simular espectro com múltiplas bandas
    baseline = 2000
    intensity = np.full_like(wavenumbers, baseline, dtype=float)
    
    # Bandas principais
    bands_example = [
        {'center': 1547, 'amplitude': 8000, 'width': 15, 'shape': 0.3},
        {'center': 1566, 'amplitude': 6000, 'width': 12, 'shape': 0.1},
        {'center': 1580, 'amplitude': 12000, 'width': 18, 'shape': 0.0},
        {'center': 1604, 'amplitude': 10000, 'width': 16, 'shape': 0.2},
        {'center': 1620, 'amplitude': 15000, 'width': 20, 'shape': 0.4},
        {'center': 1632, 'amplitude': 7000, 'width': 13, 'shape': 0.0},
        {'center': 1667, 'amplitude': 5000, 'width': 22, 'shape': 0.6}
    ]
    
    for band in bands_example:
        intensity += mixed_function(wavenumbers, band['center'], 
                                  band['amplitude'], band['width'], band['shape'])
    
    # Adicionar ruído
    noise = np.random.normal(0, 300, len(wavenumbers))
    intensity += noise
    intensity = np.maximum(intensity, 0)  # Garantir valores positivos
    
    return pd.DataFrame({
        'wavenumber': wavenumbers,
        'intensity': intensity
    })

def process_uploaded_file(uploaded_file):
    """Processa arquivo Excel/CSV carregado"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Assumir que as duas primeiras colunas são wavenumber e intensity
        df.columns = ['wavenumber', 'intensity'] + list(df.columns[2:])
        
        # Filtrar região de interesse (1500-1700 cm⁻¹)
        df = df[(df['wavenumber'] >= 1500) & (df['wavenumber'] <= 1700)]
        
        # Remover NaN
        df = df.dropna()
        
        # Ordenar por wavenumber
        df = df.sort_values('wavenumber').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return None

def calculate_fit_statistics(experimental, fitted):
    """Calcula estatísticas de ajuste"""
    residuals = experimental - fitted
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((experimental - np.mean(experimental))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    # MAE
    mae = np.mean(np.abs(residuals))
    
    return r_squared, rmse, mae, residuals

def create_export_data(spectrum_data, bands, baseline, stats):
    """Cria dados para exportação"""
    export_data = {
        'analysis_info': {
            'date': datetime.now().isoformat(),
            'software': 'Streamlit Raman Deconvolution',
            'data_points': len(spectrum_data),
            'baseline': baseline
        },
        'statistics': {
            'r_squared': stats[0],
            'rmse': stats[1],
            'mae': stats[2]
        },
        'bands': [],
        'spectrum': spectrum_data.to_dict('records')
    }
    
    for i, band in enumerate(bands):
        export_data['bands'].append({
            'id': i + 1,
            'name': band['name'],
            'center': band['center'],
            'amplitude': band['amplitude'],
            'width': band['width'],
            'shape_factor': band['shape_factor'],
            'area': calculate_area(band['amplitude'], band['width'], band['shape_factor']),
            'type': 'Gaussiana' if band['shape_factor'] == 0 else 
                   'Lorentziana' if band['shape_factor'] == 1 else 'Mista'
        })
    
    return export_data

def main():
    st.title("🔬 Deconvolução Espectral Raman")
    st.markdown("**Região 1500-1700 cm⁻¹ com suporte a Gaussiana e Lorentziana**")
    
    # Sidebar para upload e configurações
    with st.sidebar:
        st.header("📁 Dados")
        
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Carregar arquivo Excel/CSV",
            type=['xlsx', 'xls', 'csv'],
            help="Formato: Coluna A = Raman Shift (cm⁻¹), Coluna B = Intensidade"
        )
        
        # Botão para dados de exemplo
        use_example = st.button("🧪 Usar Dados de Exemplo")
        
        st.divider()
        
        # Configurações de baseline
        st.header("⚙️ Configurações Gerais")
        baseline = st.slider("Baseline", 0, 20000, 2000, 100)
        
        # Opções de visualização
        show_individual_bands = st.checkbox("Mostrar bandas individuais", True)
        show_residuals = st.checkbox("Mostrar gráfico de resíduos", True)
    
    # Carregar dados
    if uploaded_file is not None:
        spectrum_data = process_uploaded_file(uploaded_file)
        if spectrum_data is None:
            return
        st.success(f"✅ Arquivo carregado: {len(spectrum_data)} pontos espectrais")
    elif use_example:
        spectrum_data = load_example_data()
        st.success("✅ Dados de exemplo carregados")
    else:
        st.info("👆 Carregue um arquivo Excel/CSV ou use os dados de exemplo para começar")
        return
    
    # Inicializar bandas no session state
    if 'bands' not in st.session_state:
        st.session_state.bands = [
            {'name': 'Banda 1', 'center': 1547, 'amplitude': 5000, 'width': 15, 'shape_factor': 0.0, 'color': '#ff6b6b'},
            {'name': 'Banda 2', 'center': 1566, 'amplitude': 4000, 'width': 12, 'shape_factor': 0.0, 'color': '#4ecdc4'},
            {'name': 'Banda 3', 'center': 1580, 'amplitude': 8000, 'width': 18, 'shape_factor': 0.0, 'color': '#45b7d1'},
            {'name': 'Banda 4', 'center': 1604, 'amplitude': 6000, 'width': 16, 'shape_factor': 0.0, 'color': '#96ceb4'},
            {'name': 'Banda 5', 'center': 1620, 'amplitude': 5000, 'width': 14, 'shape_factor': 0.0, 'color': '#ffeaa7'},
            {'name': 'Banda 6', 'center': 1632, 'amplitude': 4500, 'width': 13, 'shape_factor': 0.0, 'color': '#dda0dd'},
            {'name': 'Banda 7', 'center': 1667, 'amplitude': 3000, 'width': 20, 'shape_factor': 0.0, 'color': '#98d8c8'}
        ]
    
    # Ajustar limites baseados nos dados
    max_intensity = spectrum_data['intensity'].max()
    max_amplitude = max(50000, int(max_intensity * 1.5))
    
    # Layout principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎛️ Controles das Bandas")
        
        # Controles para cada banda
        for i, band in enumerate(st.session_state.bands):
            with st.expander(f"{band['name']} - {band['color']}", expanded=False):
                
                # Centro
                center = st.slider(
                    f"Centro (cm⁻¹)", 
                    1500.0, 1700.0, 
                    float(band['center']), 
                    0.5, 
                    key=f"center_{i}"
                )
                
                # Amplitude
                amplitude = st.slider(
                    f"Amplitude", 
                    0, max_amplitude, 
                    int(band['amplitude']), 
                    100,
                    key=f"amplitude_{i}"
                )
                
                # FWHM
                width = st.slider(
                    f"FWHM", 
                    3.0, 80.0, 
                    float(band['width']), 
                    0.5,
                    key=f"width_{i}"
                )
                
                # Fator de forma (0 = Gaussiana, 1 = Lorentziana)
                shape_factor = st.slider(
                    "Tipo de curva", 
                    0.0, 1.0, 
                    float(band['shape_factor']), 
                    0.05,
                    key=f"shape_{i}",
                    help="0 = Gaussiana pura, 1 = Lorentziana pura"
                )
                
                # Mostrar tipo de curva
                if shape_factor == 0:
                    curve_type = "Gaussiana"
                elif shape_factor == 1:
                    curve_type = "Lorentziana"
                else:
                    curve_type = f"Mista (G: {1-shape_factor:.2f}, L: {shape_factor:.2f})"
                
                st.caption(f"Tipo: {curve_type}")
                
                # Atualizar banda
                st.session_state.bands[i].update({
                    'center': center,
                    'amplitude': amplitude,
                    'width': width,
                    'shape_factor': shape_factor
                })
    
    with col2:
        st.header("📊 Espectro e Deconvolução")
        
        # Calcular ajuste
        wavenumbers = spectrum_data['wavenumber'].values
        experimental = spectrum_data['intensity'].values
        
        # Calcular contribuição de cada banda
        fitted = np.full_like(wavenumbers, baseline, dtype=float)
        band_contributions = []
        
        for band in st.session_state.bands:
            contribution = mixed_function(
                wavenumbers, 
                band['center'], 
                band['amplitude'], 
                band['width'], 
                band['shape_factor']
            )
            fitted += contribution
            band_contributions.append(contribution)
        
        # Calcular estatísticas
        r_squared, rmse, mae, residuals = calculate_fit_statistics(experimental, fitted)
        
        # Criar gráfico principal
        fig = go.Figure()
        
        # Dados experimentais
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=experimental,
            mode='lines',
            name='Experimental',
            line=dict(color='black', width=2)
        ))
        
        # Ajuste total
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=fitted,
            mode='lines',
            name='Ajuste Total',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=np.full_like(wavenumbers, baseline),
            mode='lines',
            name='Baseline',
            line=dict(color='gray', width=1, dash='dot')
        ))
        
        # Bandas individuais
        if show_individual_bands:
            for i, (band, contribution) in enumerate(zip(st.session_state.bands, band_contributions)):
                fig.add_trace(go.Scatter(
                    x=wavenumbers,
                    y=baseline + contribution,
                    mode='lines',
                    name=band['name'],
                    line=dict(color=band['color'], width=2),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Deconvolução Espectral",
            xaxis_title="Raman Shift (cm⁻¹)",
            yaxis_title="Intensidade",
            height=500,
            legend=dict(x=1.02, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de resíduos
        if show_residuals:
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=wavenumbers,
                y=residuals,
                mode='lines+markers',
                name='Resíduos',
                line=dict(color='blue', width=1),
                marker=dict(size=3)
            ))
            
            # Linha zero
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_residuals.update_layout(
                title="Resíduos (Experimental - Ajuste)",
                xaxis_title="Raman Shift (cm⁻¹)",
                yaxis_title="Resíduo",
                height=300
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Estatísticas e resultados
    st.header("📈 Resultados da Deconvolução")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{r_squared:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("MAE", f"{mae:.2f}")
    with col4:
        st.metric("Pontos", len(spectrum_data))
    
    # Tabela de resultados
    st.subheader("📋 Parâmetros das Bandas")
    
    results_data = []
    for i, band in enumerate(st.session_state.bands):
        area = calculate_area(band['amplitude'], band['width'], band['shape_factor'])
        curve_type = "Gaussiana" if band['shape_factor'] == 0 else \
                    "Lorentziana" if band['shape_factor'] == 1 else "Mista"
        
        results_data.append({
            'Banda': band['name'],
            'Centro (cm⁻¹)': f"{band['center']:.1f}",
            'Amplitude': f"{band['amplitude']:,.0f}",
            'FWHM': f"{band['width']:.1f}",
            'Tipo': curve_type,
            'Fator Forma': f"{band['shape_factor']:.2f}",
            'Área (aprox.)': f"{area:,.0f}"
        })
    
    st.dataframe(pd.DataFrame(results_data), use_container_width=True)
    
    # Exportação
    st.header("💾 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exportar JSON"):
            export_data = create_export_data(
                spectrum_data.assign(fitted=fitted, residual=residuals),
                st.session_state.bands,
                baseline,
                (r_squared, rmse, mae)
            )
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar JSON",
                data=json_str,
                file_name=f"raman_deconvolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📄 Exportar CSV"):
            export_df = spectrum_data.copy()
            export_df['fitted'] = fitted
            export_df['residual'] = residuals
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=f"raman_spectrum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📋 Exportar Relatório"):
            report = f"""
RELATÓRIO DE DECONVOLUÇÃO RAMAN
===============================

Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Região: 1500-1700 cm⁻¹
Pontos de dados: {len(spectrum_data)}

ESTATÍSTICAS DE AJUSTE:
R² = {r_squared:.4f}
RMSE = {rmse:.2f}
MAE = {mae:.2f}
Baseline = {baseline}

PARÂMETROS DAS BANDAS:
"""
            for i, band in enumerate(st.session_state.bands):
                area = calculate_area(band['amplitude'], band['width'], band['shape_factor'])
                curve_type = "Gaussiana" if band['shape_factor'] == 0 else \
                            "Lorentziana" if band['shape_factor'] == 1 else "Mista"
                
                report += f"""
{band['name']}:
  Centro: {band['center']:.1f} cm⁻¹
  Amplitude: {band['amplitude']:,.0f}
  FWHM: {band['width']:.1f}
  Tipo: {curve_type}
  Fator de forma: {band['shape_factor']:.2f}
  Área: {area:,.0f}
"""
            
            st.download_button(
                label="📥 Baixar Relatório",
                data=report,
                file_name=f"raman_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()


