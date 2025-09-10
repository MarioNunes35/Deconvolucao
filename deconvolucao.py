import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
import io
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Deconvolução Espectral Avançada",
    page_icon="📊",
    layout="wide"
)

# Funções de forma de linha
def gaussian(x, amplitude, center, width):
    """Função Gaussiana"""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def lorentzian(x, amplitude, center, width):
    """Função Lorentziana"""
    return amplitude * width**2 / ((x - center)**2 + width**2)

def voigt(x, amplitude, center, width_g, width_l):
    """Função Voigt (convolução de Gaussiana e Lorentziana)"""
    # Aproximação simples da função Voigt
    return 0.5 * gaussian(x, amplitude, center, width_g) + 0.5 * lorentzian(x, amplitude, center, width_l)

def pseudo_voigt(x, amplitude, center, width, fraction):
    """Função Pseudo-Voigt (combinação linear de Gaussiana e Lorentziana)"""
    return fraction * lorentzian(x, amplitude, center, width) + (1 - fraction) * gaussian(x, amplitude, center, width)

def asymmetric_gaussian(x, amplitude, center, width_left, width_right):
    """Gaussiana Assimétrica"""
    result = np.zeros_like(x)
    left_mask = x < center
    right_mask = x >= center
    
    if np.any(left_mask):
        result[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / width_left) ** 2)
    if np.any(right_mask):
        result[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / width_right) ** 2)
    
    return result

def pearson_vii(x, amplitude, center, width, shape):
    """Função Pearson VII"""
    return amplitude / (1 + ((x - center) / width) ** 2) ** shape

# Classe principal para deconvolução
class SpectralDeconvolution:
    def __init__(self):
        self.peak_models = {
            'Gaussiana': ('gaussian', ['Amplitude', 'Centro', 'Largura']),
            'Lorentziana': ('lorentzian', ['Amplitude', 'Centro', 'Largura']),
            'Voigt': ('voigt', ['Amplitude', 'Centro', 'Largura G', 'Largura L']),
            'Pseudo-Voigt': ('pseudo_voigt', ['Amplitude', 'Centro', 'Largura', 'Fração L']),
            'Gaussiana Assimétrica': ('asymmetric_gaussian', ['Amplitude', 'Centro', 'Largura Esq', 'Largura Dir']),
            'Pearson VII': ('pearson_vii', ['Amplitude', 'Centro', 'Largura', 'Forma'])
        }
        
    def create_composite_function(self, peak_params: List[Dict]) -> callable:
        """Cria função composta com múltiplos picos"""
        def composite(x, *params):
            result = np.zeros_like(x)
            param_idx = 0
            
            for peak_info in peak_params:
                model_type = peak_info['type']
                n_params = len(self.peak_models[model_type][1])
                peak_params_values = params[param_idx:param_idx + n_params]
                
                if model_type == 'Gaussiana':
                    result += gaussian(x, *peak_params_values)
                elif model_type == 'Lorentziana':
                    result += lorentzian(x, *peak_params_values)
                elif model_type == 'Voigt':
                    result += voigt(x, *peak_params_values)
                elif model_type == 'Pseudo-Voigt':
                    result += pseudo_voigt(x, *peak_params_values)
                elif model_type == 'Gaussiana Assimétrica':
                    result += asymmetric_gaussian(x, *peak_params_values)
                elif model_type == 'Pearson VII':
                    result += pearson_vii(x, *peak_params_values)
                
                param_idx += n_params
            
            return result
        
        return composite
    
    def fit_spectrum(self, x, y, peak_params, method='leastsq'):
        """Ajusta o espectro com os parâmetros fornecidos"""
        composite_func = self.create_composite_function(peak_params)
        
        # Preparar parâmetros iniciais e limites
        p0 = []
        bounds_lower = []
        bounds_upper = []
        
        for peak in peak_params:
            for param_value, param_bounds in zip(peak['params'], peak['bounds']):
                p0.append(param_value)
                bounds_lower.append(param_bounds[0])
                bounds_upper.append(param_bounds[1])
        
        try:
            if method == 'leastsq':
                popt, pcov = curve_fit(composite_func, x, y, p0=p0, 
                                      bounds=(bounds_lower, bounds_upper),
                                      maxfev=5000)
            else:  # differential_evolution
                def objective(params):
                    return np.sum((y - composite_func(x, *params))**2)
                
                bounds = list(zip(bounds_lower, bounds_upper))
                result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
                popt = result.x
                pcov = None
            
            return popt, pcov
        except Exception as e:
            st.error(f"Erro no ajuste: {str(e)}")
            return p0, None

# Inicialização do estado da sessão
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'peaks' not in st.session_state:
    st.session_state.peaks = []
if 'fit_results' not in st.session_state:
    st.session_state.fit_results = None
if 'deconvolver' not in st.session_state:
    st.session_state.deconvolver = SpectralDeconvolution()

# Interface principal
st.title("🔬 Deconvolução Espectral Avançada")
st.markdown("---")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Upload de dados
    st.subheader("📁 Carregar Dados")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV/TXT/Excel",
        type=['csv', 'txt', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Seleção de colunas
            st.subheader("📊 Seleção de Colunas")
            x_col = st.selectbox("Coluna X (comprimento de onda/energia)", df.columns)
            y_col = st.selectbox("Coluna Y (intensidade)", df.columns)
            
            st.session_state.x_data = df[x_col].values
            st.session_state.y_data = df[y_col].values
            
            st.success(f"✅ Dados carregados: {len(df)} pontos")
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
    
    # Opção para usar dados de exemplo
    if st.button("🎲 Usar Dados de Exemplo"):
        np.random.seed(42)
        x = np.linspace(0, 100, 500)
        y = (gaussian(x, 100, 30, 5) + 
             gaussian(x, 80, 45, 8) + 
             lorentzian(x, 60, 60, 4) +
             gaussian(x, 40, 75, 6) +
             np.random.normal(0, 2, len(x)))
        
        st.session_state.x_data = x
        st.session_state.y_data = y
        st.session_state.data_loaded = True
        st.success("✅ Dados de exemplo carregados")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        
        # Detecção automática de picos
        st.subheader("🔍 Detecção de Picos")
        
        col1, col2 = st.columns(2)
        with col1:
            min_height = st.number_input(
                "Altura mínima (%)", 
                min_value=1, max_value=100, value=10
            )
        with col2:
            min_distance = st.number_input(
                "Distância mínima", 
                min_value=1, max_value=100, value=5
            )
        
        if st.button("🔎 Detectar Picos Automaticamente"):
            height_threshold = np.max(st.session_state.y_data) * min_height / 100
            peaks_idx, properties = find_peaks(
                st.session_state.y_data, 
                height=height_threshold,
                distance=min_distance
            )
            
            st.session_state.peaks = []
            for idx in peaks_idx:
                peak_x = st.session_state.x_data[idx]
                peak_y = st.session_state.y_data[idx]
                
                # Estimar largura
                half_max = peak_y / 2
                left_idx = idx
                right_idx = idx
                
                while left_idx > 0 and st.session_state.y_data[left_idx] > half_max:
                    left_idx -= 1
                while right_idx < len(st.session_state.y_data) - 1 and st.session_state.y_data[right_idx] > half_max:
                    right_idx += 1
                
                width = st.session_state.x_data[right_idx] - st.session_state.x_data[left_idx]
                
                st.session_state.peaks.append({
                    'type': 'Gaussiana',
                    'params': [peak_y, peak_x, width/2],
                    'bounds': [
                        (0, peak_y * 2),
                        (peak_x - width, peak_x + width),
                        (0.1, width * 2)
                    ]
                })
            
            st.success(f"✅ {len(peaks_idx)} picos detectados")
        
        # Configurações de ajuste
        st.markdown("---")
        st.subheader("🎯 Método de Ajuste")
        fit_method = st.selectbox(
            "Algoritmo de otimização",
            ['leastsq', 'differential_evolution']
        )

# Área principal
if st.session_state.data_loaded:
    # Tabs para organizar o conteúdo
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualização", "🎛️ Configurar Picos", "📊 Resultados", "💾 Exportar"])
    
    with tab1:
        # Plotagem interativa
        fig = go.Figure()
        
        # Dados originais
        fig.add_trace(go.Scatter(
            x=st.session_state.x_data,
            y=st.session_state.y_data,
            mode='lines',
            name='Dados Originais',
            line=dict(color='black', width=2)
        ))
        
        # Se houver ajuste, plotar
        if st.session_state.fit_results is not None and len(st.session_state.peaks) > 0:
            # Curva ajustada total
            composite_func = st.session_state.deconvolver.create_composite_function(st.session_state.peaks)
            y_fit = composite_func(st.session_state.x_data, *st.session_state.fit_results)
            
            fig.add_trace(go.Scatter(
                x=st.session_state.x_data,
                y=y_fit,
                mode='lines',
                name='Ajuste Total',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Componentes individuais
            param_idx = 0
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
            
            for i, peak in enumerate(st.session_state.peaks):
                model_type = peak['type']
                n_params = len(st.session_state.deconvolver.peak_models[model_type][1])
                peak_params = st.session_state.fit_results[param_idx:param_idx + n_params]
                
                if model_type == 'Gaussiana':
                    y_component = gaussian(st.session_state.x_data, *peak_params)
                elif model_type == 'Lorentziana':
                    y_component = lorentzian(st.session_state.x_data, *peak_params)
                elif model_type == 'Voigt':
                    y_component = voigt(st.session_state.x_data, *peak_params)
                elif model_type == 'Pseudo-Voigt':
                    y_component = pseudo_voigt(st.session_state.x_data, *peak_params)
                elif model_type == 'Gaussiana Assimétrica':
                    y_component = asymmetric_gaussian(st.session_state.x_data, *peak_params)
                elif model_type == 'Pearson VII':
                    y_component = pearson_vii(st.session_state.x_data, *peak_params)
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.x_data,
                    y=y_component,
                    mode='lines',
                    name=f'Pico {i+1} ({model_type})',
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    fill='tozeroy',
                    fillcolor=colors[i % len(colors)],
                    opacity=0.3
                ))
                
                param_idx += n_params
            
            # Resíduos
            residuals = st.session_state.y_data - y_fit
            fig.add_trace(go.Scatter(
                x=st.session_state.x_data,
                y=residuals,
                mode='lines',
                name='Resíduos',
                line=dict(color='gray', width=1),
                yaxis='y2'
            ))
        
        # Layout do gráfico
        fig.update_layout(
            title='Deconvolução Espectral',
            xaxis_title='X',
            yaxis_title='Intensidade',
            yaxis2=dict(
                title='Resíduos',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=600,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de qualidade do ajuste
        if st.session_state.fit_results is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            composite_func = st.session_state.deconvolver.create_composite_function(st.session_state.peaks)
            y_fit = composite_func(st.session_state.x_data, *st.session_state.fit_results)
            residuals = st.session_state.y_data - y_fit
            
            r_squared = 1 - np.sum(residuals**2) / np.sum((st.session_state.y_data - np.mean(st.session_state.y_data))**2)
            rmse = np.sqrt(np.mean(residuals**2))
            chi_squared = np.sum(residuals**2 / y_fit) if np.all(y_fit > 0) else np.inf
            
            col1.metric("R²", f"{r_squared:.4f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("χ²", f"{chi_squared:.2f}")
            col4.metric("N° Picos", len(st.session_state.peaks))
    
    with tab2:
        st.subheader("🎛️ Configuração Manual dos Picos")
        
        # Adicionar novo pico
        with st.expander("➕ Adicionar Novo Pico", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Métricas de Qualidade:**")
                
                r_squared = 1 - np.sum(residuals**2) / np.sum((st.session_state.y_data - np.mean(st.session_state.y_data))**2)
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
                chi_squared = np.sum(residuals**2 / np.abs(y_fit)) if np.all(y_fit != 0) else np.inf
                
                metrics_df = pd.DataFrame({
                    'Métrica': ['R²', 'RMSE', 'MAE', 'χ² reduzido'],
                    'Valor': [
                        f"{r_squared:.6f}",
                        f"{rmse:.4f}",
                        f"{mae:.4f}",
                        f"{chi_squared/len(st.session_state.x_data):.4f}"
                    ]
                })
                st.dataframe(metrics_df, hide_index=True)
            
            with col2:
                st.markdown("**Análise de Resíduos:**")
                
                # Histograma de resíduos
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Resíduos',
                    marker_color='blue',
                    opacity=0.7
                ))
                
                fig_hist.update_layout(
                    title='Distribuição dos Resíduos',
                    xaxis_title='Resíduo',
                    yaxis_title='Frequência',
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Teste de normalidade dos resíduos
                from scipy import stats
                _, p_value = stats.normaltest(residuals)
                
                if p_value > 0.05:
                    st.success(f"✅ Resíduos normalmente distribuídos (p={p_value:.4f})")
                else:
                    st.warning(f"⚠️ Resíduos podem não ser normais (p={p_value:.4f})")
    
    with tab4:
        st.subheader("💾 Exportar Resultados")
        
        if st.session_state.fit_results is not None and len(st.session_state.peaks) > 0:
            # Preparar dados para exportação
            composite_func = st.session_state.deconvolver.create_composite_function(st.session_state.peaks)
            y_fit = composite_func(st.session_state.x_data, *st.session_state.fit_results)
            
            # Dados completos com componentes individuais
            export_data = {
                'X': st.session_state.x_data,
                'Y_Original': st.session_state.y_data,
                'Y_Ajuste': y_fit,
                'Resíduos': st.session_state.y_data - y_fit
            }
            
            # Adicionar componentes individuais
            param_idx = 0
            for i, peak in enumerate(st.session_state.peaks):
                model_type = peak['type']
                n_params = len(st.session_state.deconvolver.peak_models[model_type][1])
                peak_params = st.session_state.fit_results[param_idx:param_idx + n_params]
                
                if model_type == 'Gaussiana':
                    y_component = gaussian(st.session_state.x_data, *peak_params)
                elif model_type == 'Lorentziana':
                    y_component = lorentzian(st.session_state.x_data, *peak_params)
                elif model_type == 'Voigt':
                    y_component = voigt(st.session_state.x_data, *peak_params)
                elif model_type == 'Pseudo-Voigt':
                    y_component = pseudo_voigt(st.session_state.x_data, *peak_params)
                elif model_type == 'Gaussiana Assimétrica':
                    y_component = asymmetric_gaussian(st.session_state.x_data, *peak_params)
                elif model_type == 'Pearson VII':
                    y_component = pearson_vii(st.session_state.x_data, *peak_params)
                
                export_data[f'Pico_{i+1}_{model_type}'] = y_component
                param_idx += n_params
            
            df_export = pd.DataFrame(export_data)
            
            # Opções de exportação
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Exportar Dados Completos**")
                
                # CSV
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="deconvolucao_resultados.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("**📈 Exportar Parâmetros**")
                
                # Preparar tabela de parâmetros
                params_data = []
                param_idx = 0
                
                for i, peak in enumerate(st.session_state.peaks):
                    model_type = peak['type']
                    param_names = st.session_state.deconvolver.peak_models[model_type][1]
                    n_params = len(param_names)
                    peak_params = st.session_state.fit_results[param_idx:param_idx + n_params]
                    
                    params_dict = {
                        'Pico': f"Pico {i+1}",
                        'Tipo': model_type
                    }
                    
                    for param_name, param_value in zip(param_names, peak_params):
                        params_dict[param_name] = param_value
                    
                    params_data.append(params_dict)
                    param_idx += n_params
                
                df_params = pd.DataFrame(params_data)
                csv_params = df_params.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Parâmetros",
                    data=csv_params,
                    file_name="parametros_ajuste.csv",
                    mime="text/csv"
                )
            
            with col3:
                st.markdown("**📄 Exportar Relatório**")
                
                # Gerar relatório em texto
                report = f"""RELATÓRIO DE DECONVOLUÇÃO ESPECTRAL
=====================================
Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ESTATÍSTICAS DO AJUSTE
----------------------
R²: {1 - np.sum((st.session_state.y_data - y_fit)**2) / np.sum((st.session_state.y_data - np.mean(st.session_state.y_data))**2):.6f}
RMSE: {np.sqrt(np.mean((st.session_state.y_data - y_fit)**2)):.4f}
MAE: {np.mean(np.abs(st.session_state.y_data - y_fit)):.4f}
Número de picos: {len(st.session_state.peaks)}
Número de pontos: {len(st.session_state.x_data)}

PARÂMETROS DOS PICOS
--------------------
"""
                param_idx = 0
                total_area = 0
                areas = []
                
                for i, peak in enumerate(st.session_state.peaks):
                    model_type = peak['type']
                    param_names = st.session_state.deconvolver.peak_models[model_type][1]
                    n_params = len(param_names)
                    peak_params = st.session_state.fit_results[param_idx:param_idx + n_params]
                    
                    report += f"\nPico {i+1} ({model_type}):\n"
                    
                    for param_name, param_value in zip(param_names, peak_params):
                        report += f"  {param_name}: {param_value:.4f}\n"
                    
                    # Calcular área
                    if model_type == 'Gaussiana':
                        area = peak_params[0] * peak_params[2] * np.sqrt(2 * np.pi)
                    elif model_type == 'Lorentziana':
                        area = peak_params[0] * peak_params[2] * np.pi
                    else:
                        if model_type == 'Voigt':
                            y_component = voigt(st.session_state.x_data, *peak_params)
                        elif model_type == 'Pseudo-Voigt':
                            y_component = pseudo_voigt(st.session_state.x_data, *peak_params)
                        elif model_type == 'Gaussiana Assimétrica':
                            y_component = asymmetric_gaussian(st.session_state.x_data, *peak_params)
                        elif model_type == 'Pearson VII':
                            y_component = pearson_vii(st.session_state.x_data, *peak_params)
                        area = np.trapz(y_component, st.session_state.x_data)
                    
                    areas.append(area)
                    total_area += area
                    report += f"  Área: {area:.4f}\n"
                    
                    param_idx += n_params
                
                report += "\nCONTRIBUIÇÃO PERCENTUAL\n"
                report += "-----------------------\n"
                for i, area in enumerate(areas):
                    report += f"Pico {i+1}: {(area/total_area)*100:.2f}%\n"
                
                st.download_button(
                    label="📥 Download Relatório",
                    data=report,
                    file_name="relatorio_deconvolucao.txt",
                    mime="text/plain"
                )
            
            # Exportar Excel com múltiplas abas
            st.markdown("---")
            st.markdown("**📑 Exportar Excel Completo (múltiplas abas)**")
            
            # Criar arquivo Excel em memória
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Aba 1: Dados completos
                df_export.to_excel(writer, sheet_name='Dados_Completos', index=False)
                
                # Aba 2: Parâmetros
                df_params.to_excel(writer, sheet_name='Parametros', index=False)
                
                # Aba 3: Estatísticas
                stats_data = {
                    'Métrica': ['R²', 'RMSE', 'MAE', 'χ² reduzido', 'N° Picos', 'N° Pontos'],
                    'Valor': [
                        1 - np.sum((st.session_state.y_data - y_fit)**2) / np.sum((st.session_state.y_data - np.mean(st.session_state.y_data))**2),
                        np.sqrt(np.mean((st.session_state.y_data - y_fit)**2)),
                        np.mean(np.abs(st.session_state.y_data - y_fit)),
                        np.sum((st.session_state.y_data - y_fit)**2 / np.abs(y_fit)) / len(st.session_state.x_data) if np.all(y_fit != 0) else np.inf,
                        len(st.session_state.peaks),
                        len(st.session_state.x_data)
                    ]
                }
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Estatisticas', index=False)
                
                # Aba 4: Áreas
                areas_data = []
                param_idx = 0
                for i, peak in enumerate(st.session_state.peaks):
                    model_type = peak['type']
                    n_params = len(st.session_state.deconvolver.peak_models[model_type][1])
                    peak_params = st.session_state.fit_results[param_idx:param_idx + n_params]
                    
                    if model_type == 'Gaussiana':
                        area = peak_params[0] * peak_params[2] * np.sqrt(2 * np.pi)
                    elif model_type == 'Lorentziana':
                        area = peak_params[0] * peak_params[2] * np.pi
                    else:
                        if model_type == 'Voigt':
                            y_component = voigt(st.session_state.x_data, *peak_params)
                        elif model_type == 'Pseudo-Voigt':
                            y_component = pseudo_voigt(st.session_state.x_data, *peak_params)
                        elif model_type == 'Gaussiana Assimétrica':
                            y_component = asymmetric_gaussian(st.session_state.x_data, *peak_params)
                        elif model_type == 'Pearson VII':
                            y_component = pearson_vii(st.session_state.x_data, *peak_params)
                        area = np.trapz(y_component, st.session_state.x_data)
                    
                    areas_data.append({
                        'Pico': f"Pico {i+1}",
                        'Tipo': model_type,
                        'Área': area,
                        'Área (%)': (area/total_area)*100 if total_area > 0 else 0
                    })
                    param_idx += n_params
                
                df_areas = pd.DataFrame(areas_data)
                df_areas.to_excel(writer, sheet_name='Areas', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel Completo",
                data=output,
                file_name="deconvolucao_completo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        else:
            st.warning("⚠️ Realize o ajuste primeiro para poder exportar os resultados.")

else:
    # Página inicial quando não há dados carregados
    st.info("👈 Por favor, carregue seus dados usando o menu lateral ou use os dados de exemplo para começar.")
    
    st.markdown("""
    ### 📋 Instruções de Uso:
    
    1. **Carregar Dados**: Use o menu lateral para carregar um arquivo CSV, TXT ou Excel com seus dados espectrais
    2. **Detectar Picos**: Use a detecção automática ou adicione picos manualmente
    3. **Configurar Picos**: Ajuste os parâmetros e tipos de cada pico
    4. **Realizar Ajuste**: Execute o algoritmo de otimização
    5. **Visualizar**: Analise os resultados em tempo real
    6. **Exportar**: Baixe os dados deconvoluídos e parâmetros
    
    ### 🎯 Recursos Principais:
    
    - ✅ Múltiplos modelos de pico (Gaussiana, Lorentziana, Voigt, etc.)
    - ✅ Ajuste em tempo real com visualização interativa
    - ✅ Detecção automática de picos
    - ✅ Customização completa de parâmetros
    - ✅ Exportação em múltiplos formatos
    - ✅ Análise estatística do ajuste
    - ✅ Interface intuitiva e responsiva
    """)
    
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ usando Streamlit e SciPy")
                peak_type = st.selectbox(
                    "Tipo de Pico",
                    list(st.session_state.deconvolver.peak_models.keys()),
                    key="new_peak_type"
                )
            
            with col2:
                if st.button("➕ Adicionar Pico"):
                    # Valores padrão baseados nos dados
                    x_center = np.mean(st.session_state.x_data)
                    y_max = np.max(st.session_state.y_data)
                    x_range = np.max(st.session_state.x_data) - np.min(st.session_state.x_data)
                    
                    if peak_type == 'Gaussiana':
                        params = [y_max/2, x_center, x_range/20]
                        bounds = [(0, y_max*2), (np.min(st.session_state.x_data), np.max(st.session_state.x_data)), (0.1, x_range)]
                    elif peak_type == 'Lorentziana':
                        params = [y_max/2, x_center, x_range/20]
                        bounds = [(0, y_max*2), (np.min(st.session_state.x_data), np.max(st.session_state.x_data)), (0.1, x_range)]
                    elif peak_type == 'Voigt':
                        params = [y_max/2, x_center, x_range/20, x_range/20]
                        bounds = [(0, y_max*2), (np.min(st.session_state.x_data), np.max(st.session_state.x_data)), 
                                (0.1, x_range), (0.1, x_range)]
                    elif peak_type == 'Pseudo-Voigt':
                        params = [y_max/2, x_center, x_range/20, 0.5]
                        bounds = [(0, y_max*2), (np.min(st.session_state.x_data), np.max(st.session_state.x_data)), 
                                (0.1, x_range), (0, 1)]
                    elif peak_type == 'Gaussiana Assimétrica':
                        params = [y_max/2, x_center, x_range/20, x_range/20]
                        bounds = [(0, y_max*2), (np.min(st.session_state.x_data), np.max(st.session_state.x_data)), 
                                (0.1, x_range), (0.1, x_range)]
                    elif peak_type == 'Pearson VII':
                        params = [y_max/2, x_center, x_range/20, 1.5]
                        bounds = [(0, y_max*2), (np.min(st.session_state.x_data), np.max(st.session_state.x_data)), 
                                (0.1, x_range), (0.5, 10)]
                    
                    st.session_state.peaks.append({
                        'type': peak_type,
                        'params': params,
                        'bounds': bounds
                    })
                    st.success(f"✅ Pico {peak_type} adicionado")
                    st.rerun()
        
        # Editar picos existentes
        if len(st.session_state.peaks) > 0:
            st.markdown("---")
            st.subheader("📝 Editar Picos Existentes")
            
            for i, peak in enumerate(st.session_state.peaks):
                with st.expander(f"Pico {i+1}: {peak['type']}", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**Parâmetros Atuais:**")
                        param_names = st.session_state.deconvolver.peak_models[peak['type']][1]
                        
                        new_params = []
                        for j, (param_name, param_value, param_bounds) in enumerate(zip(param_names, peak['params'], peak['bounds'])):
                            new_value = st.slider(
                                param_name,
                                min_value=float(param_bounds[0]),
                                max_value=float(param_bounds[1]),
                                value=float(param_value),
                                key=f"peak_{i}_param_{j}",
                                format="%.3f"
                            )
                            new_params.append(new_value)
                        
                        st.session_state.peaks[i]['params'] = new_params
                    
                    with col2:
                        st.markdown("**Limites dos Parâmetros:**")
                        new_bounds = []
                        for j, (param_name, param_bounds) in enumerate(zip(param_names, peak['bounds'])):
                            col_min, col_max = st.columns(2)
                            with col_min:
                                min_val = st.number_input(
                                    f"Min {param_name}",
                                    value=float(param_bounds[0]),
                                    key=f"peak_{i}_bound_min_{j}",
                                    format="%.3f"
                                )
                            with col_max:
                                max_val = st.number_input(
                                    f"Max {param_name}",
                                    value=float(param_bounds[1]),
                                    key=f"peak_{i}_bound_max_{j}",
                                    format="%.3f"
                                )
                            new_bounds.append((min_val, max_val))
                        
                        st.session_state.peaks[i]['bounds'] = new_bounds
                    
                    with col3:
                        st.markdown("**Ações:**")
                        if st.button(f"🗑️ Remover", key=f"remove_peak_{i}"):
                            st.session_state.peaks.pop(i)
                            st.rerun()
                        
                        # Mudar tipo de pico
                        new_type = st.selectbox(
                            "Mudar tipo",
                            list(st.session_state.deconvolver.peak_models.keys()),
                            index=list(st.session_state.deconvolver.peak_models.keys()).index(peak['type']),
                            key=f"change_type_{i}"
                        )
                        
                        if new_type != peak['type']:
                            # Ajustar número de parâmetros se necessário
                            old_n_params = len(st.session_state.deconvolver.peak_models[peak['type']][1])
                            new_n_params = len(st.session_state.deconvolver.peak_models[new_type][1])
                            
                            if new_n_params > old_n_params:
                                # Adicionar parâmetros extras
                                for _ in range(new_n_params - old_n_params):
                                    st.session_state.peaks[i]['params'].append(1.0)
                                    st.session_state.peaks[i]['bounds'].append((0.1, 10.0))
                            elif new_n_params < old_n_params:
                                # Remover parâmetros extras
                                st.session_state.peaks[i]['params'] = st.session_state.peaks[i]['params'][:new_n_params]
                                st.session_state.peaks[i]['bounds'] = st.session_state.peaks[i]['bounds'][:new_n_params]
                            
                            st.session_state.peaks[i]['type'] = new_type
                            st.rerun()
        
        # Botão para realizar ajuste
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Realizar Ajuste", type="primary", disabled=len(st.session_state.peaks) == 0):
                with st.spinner("Ajustando..."):
                    fit_results, pcov = st.session_state.deconvolver.fit_spectrum(
                        st.session_state.x_data,
                        st.session_state.y_data,
                        st.session_state.peaks,
                        method=fit_method
                    )
                    
                    st.session_state.fit_results = fit_results
                    
                    # Atualizar parâmetros dos picos com os resultados do ajuste
                    param_idx = 0
                    for i, peak in enumerate(st.session_state.peaks):
                        n_params = len(st.session_state.deconvolver.peak_models[peak['type']][1])
                        st.session_state.peaks[i]['params'] = list(fit_results[param_idx:param_idx + n_params])
                        param_idx += n_params
                    
                    st.success("✅ Ajuste concluído!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Resetar Picos"):
                st.session_state.peaks = []
                st.session_state.fit_results = None
                st.rerun()
    
    with tab3:
        st.subheader("📊 Resultados Detalhados")
        
        if st.session_state.fit_results is not None and len(st.session_state.peaks) > 0:
            # Tabela de parâmetros
            results_data = []
            param_idx = 0
            
            for i, peak in enumerate(st.session_state.peaks):
                model_type = peak['type']
                param_names = st.session_state.deconvolver.peak_models[model_type][1]
                n_params = len(param_names)
                peak_params = st.session_state.fit_results[param_idx:param_idx + n_params]
                
                # Calcular área do pico
                if model_type == 'Gaussiana':
                    area = peak_params[0] * peak_params[2] * np.sqrt(2 * np.pi)
                elif model_type == 'Lorentziana':
                    area = peak_params[0] * peak_params[2] * np.pi
                else:
                    # Integração numérica para outros tipos
                    if model_type == 'Voigt':
                        y_component = voigt(st.session_state.x_data, *peak_params)
                    elif model_type == 'Pseudo-Voigt':
                        y_component = pseudo_voigt(st.session_state.x_data, *peak_params)
                    elif model_type == 'Gaussiana Assimétrica':
                        y_component = asymmetric_gaussian(st.session_state.x_data, *peak_params)
                    elif model_type == 'Pearson VII':
                        y_component = pearson_vii(st.session_state.x_data, *peak_params)
                    
                    area = np.trapz(y_component, st.session_state.x_data)
                
                result_dict = {
                    'Pico': f"Pico {i+1}",
                    'Tipo': model_type,
                    'Área': area,
                    'Área (%)': 0  # Será calculado depois
                }
                
                for param_name, param_value in zip(param_names, peak_params):
                    result_dict[param_name] = param_value
                
                results_data.append(result_dict)
                param_idx += n_params
            
            # Calcular porcentagem de área
            total_area = sum([r['Área'] for r in results_data])
            for r in results_data:
                r['Área (%)'] = (r['Área'] / total_area) * 100
            
            # Exibir tabela
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Estatísticas do ajuste
            st.markdown("---")
            st.subheader("📈 Estatísticas do Ajuste")
            
            composite_func = st.session_state.deconvolver.create_composite_function(st.session_state.peaks)
            y_fit = composite_func(st.session_state.x_data, *st.session_state.fit_results)
            residuals = st.session_state.y_data - y_fit
            
            col1, col2 = st.columns(2)
            
            with col1:


