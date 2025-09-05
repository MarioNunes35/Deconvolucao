#!/usr/bin/env python3
"""
Launcher Python para Aplicação de Deconvolução Raman
Versão robusta com detecção de erros e fallback offline
CONTROLES AJUSTADOS PARA ALTAS INTENSIDADES
"""

import os
import sys
import webbrowser
import threading
import time
import json
from flask import Flask, render_template_string, send_from_directory, jsonify
from werkzeug.serving import make_server
import subprocess

class RamanApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.server = None
        self.port = 8080
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            return send_from_directory('static', filename)
        
        @self.app.route('/data/<path:filename>')
        def data_files(filename):
            return send_from_directory('.', filename)
            
        @self.app.route('/health')
        def health():
            return jsonify({"status": "running", "message": "Raman Deconvolution App"})

    def get_html_template(self):
        return '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deconvolução Espectral Raman</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
    <script>
    // Fallback se o CDN principal falhar
    (function(){
        function loadFallback(){
            var s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
            s.onload = function(){ console.log('✅ XLSX fallback carregado'); };
            s.onerror = function(){ console.error('❌ Falha ao carregar XLSX'); };
            document.head.appendChild(s);
        }
        if (typeof XLSX === 'undefined') {
            // aguarda um pouco e confere de novo
            setTimeout(function(){
                if (typeof XLSX === 'undefined') loadFallback();
            }, 800);
        }
    })();
    </script>

    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #f8fafc;
            color: #334155;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #1e293b; font-size: 2.5rem; margin-bottom: 10px; }
        .header p { color: #64748b; font-size: 1.1rem; }
        .loading { 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 60vh; 
            font-size: 18px;
            flex-direction: column;
            gap: 20px;
        }
        .spinner {
            border: 4px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .controls { 
            display: grid; 
            grid-template-columns: 1fr 2fr; 
            gap: 30px; 
            margin-bottom: 30px;
        }
        .control-panel, .chart-panel { 
            background: white; 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .control-section { margin-bottom: 25px; }
        .control-section h3 { 
            color: #1e293b; 
            margin-bottom: 15px; 
            font-size: 1.2rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }
        .input-group { margin-bottom: 15px; }
        .input-group label { 
            display: block; 
            font-weight: 600; 
            margin-bottom: 5px; 
            color: #475569;
        }
        .input-group input[type="range"] { 
            width: 100%; 
            margin-bottom: 8px;
            height: 6px;
            border-radius: 3px;
            background: #e2e8f0;
            outline: none;
        }
        .input-group input[type="number"] { 
            width: 100%; 
            padding: 8px 12px; 
            border: 1px solid #d1d5db; 
            border-radius: 6px;
            font-size: 14px;
        }
        .band-control {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .band-header {
            font-weight: 600;
            margin-bottom: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            color: white;
        }
        .stats {
            background: #dbeafe;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats h4 { color: #1e40af; margin-bottom: 10px; }
        .stats div { margin-bottom: 5px; font-weight: 600; }
        .button {
            background: #10b981;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: background 0.3s;
        }
        .button:hover { background: #059669; }
        .button-secondary {
            background: #6b7280;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            flex: 1;
            transition: background 0.3s;
        }
        .button-secondary:hover { background: #4b5563; }
        .chart-container {
            width: 100%;
            height: 400px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            color: #64748b;
            position: relative;
            overflow: hidden;
        }
        .chart-container canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        .results-table {
            width: 100%;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        .table th {
            background: #f8fafc;
            font-weight: 600;
            color: #374151;
        }
        .error-message {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .success-message {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #16a34a;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 12px;
            width: 90%;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 20px;
        }
        .modal-header h3 {
            margin: 0;
            color: #1e293b;
        }
        .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #64748b;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .auto-adjust-notice {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #c2410c;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Deconvolução Espectral Raman</h1>
            <p>Região 1500-1700 cm⁻¹ - Controles Ajustados para Altas Intensidades</p>
        </div>

        <div id="loading" class="loading">
            <div style="text-align: center;">
                <h3 style="margin-bottom: 20px;">🔍 Carregar Dados Espectrais</h3>
                <p style="margin-bottom: 20px; color: #64748b;">
                    Selecione seu arquivo Excel com dados Raman (região 1500-1700 cm⁻¹)
                </p>
                
                <div id="dropzone" style="border: 2px dashed #d1d5db; padding: 30px; border-radius: 12px; margin-bottom: 20px;">
                    <input type="file" id="fileInput" accept=".xlsx,.xls,.csv" style="display: none;" onchange="handleFileSelect(event)">
                    <button class="button" onclick="document.getElementById('fileInput').click()">
                        📂 Selecionar Arquivo Excel
                    </button>
                    <p style="margin-top: 10px; font-size: 14px; color: #64748b;">
                        Ou arraste e solte seu arquivo aqui
                    </p>
                </div>
                
                <div style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <p style="font-size: 14px; color: #64748b; margin-bottom: 10px;"><strong>Formato esperado:</strong></p>
                    <ul style="font-size: 13px; color: #64748b; margin-left: 20px;">
                        <li>Coluna A: Raman Shift (cm⁻¹)</li>
                        <li>Coluna B: Intensidade</li>
                        <li>Primeira linha pode ter cabeçalhos</li>
                        <li>Dados na região 1500-1700 cm⁻¹</li>
                    </ul>
                </div>
                
                <div style="margin-top: 20px;">
                    <button class="button-secondary" onclick="loadExampleData()" style="background: #6b7280; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer;">
                        🧪 Usar Dados de Exemplo
                    </button>
                </div>
                
                <div id="loadingSpinner" style="display: none; margin-top: 20px;">
                    <div class="spinner"></div>
                    <div id="loadingStatus">Processando arquivo...</div>
                </div>
            </div>
        </div>

        <div id="app" style="display: none;">
            <div class="controls">
                <div class="control-panel">
                    <div class="control-section">
                        <h3>Controles Gerais</h3>
                        
                        <div class="auto-adjust-notice">
                            ℹ️ Controles ajustados automaticamente baseados nos dados carregados
                        </div>
                        
                        <div class="input-group">
                            <label>Baseline:</label>
                            <input type="range" id="baseline" min="0" max="15000" value="1000" step="50">
                            <input type="number" id="baselineValue" value="1000" step="50">
                        </div>

                        <div class="checkbox-group">
                            <input type="checkbox" id="showBands" checked>
                            <label for="showBands">Mostrar bandas individuais</label>
                        </div>

                        <div class="stats">
                            <h4>Estatísticas de Ajuste</h4>
                            <div>R² = <span id="rSquared">0.0000</span></div>
                            <div>RMSE = <span id="rmse">0.00</span></div>
                            <div>MAE = <span id="mae">0.00</span></div>
                            <div>Pontos = <span id="dataPoints">0</span></div>
                        </div>

                        <button class="button" onclick="showExportModal()">
                            📊 Configurar Exportação
                        </button>
                    </div>

                    <div class="control-section">
                        <h3>Parâmetros das Bandas</h3>
                        <div id="bandControls"></div>
                    </div>
                </div>

                <div class="chart-panel">
                    <h3>Espectro e Deconvolução</h3>
                    <div class="chart-container" id="mainChart">
                        Gráfico será carregado aqui
                    </div>
                    
                    <h3>Resíduos (Experimental - Ajuste)</h3>
                    <div class="chart-container" id="residualChart">
                        Gráfico de resíduos será carregado aqui
                    </div>
                </div>
            </div>

            <div class="results-table">
                <h3>Resultados da Deconvolução</h3>
                <table class="table" id="resultsTable">
                    <thead>
                        <tr>
                            <th>Banda</th>
                            <th>Centro (cm⁻¹)</th>
                            <th>Amplitude</th>
                            <th>FWHM</th>
                            <th>Área (aprox.)</th>
                        </tr>
                    </thead>
                    <tbody id="resultsTableBody">
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Modal de Exportação -->
        <div id="exportModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Configurações de Exportação</h3>
                    <button class="close-btn" onclick="hideExportModal()">&times;</button>
                </div>

                <div class="input-group">
                    <label>Formato de Exportação:</label>
                    <select id="exportFormat" style="width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 6px;">
                        <option value="json">JSON (Recomendado)</option>
                        <option value="csv">CSV (Para Excel/Origin)</option>
                        <option value="txt">TXT (Texto simples)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Dados para exportar:</label>
                    <div class="checkbox-group">
                        <input type="checkbox" id="includeBands" checked>
                        <label for="includeBands">Parâmetros das bandas</label>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="includeSpectrum" checked>
                        <label for="includeSpectrum">Dados do espectro</label>
                    </div>
                    <div class="checkbox-group">
                        <input type="checkbox" id="includeStats" checked>
                        <label for="includeStats">Estatísticas</label>
                    </div>
                </div>

                <div class="button-group">
                    <button class="button-secondary" onclick="hideExportModal()">Cancelar</button>
                    <button class="button" onclick="performExport()">📊 Exportar</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dados globais
        let spectrumData = [];
        let bands = [
            { id: 1, center: 1547, amplitude: 2000, width: 15, color: '#ff6b6b', name: 'Banda 1' },
            { id: 2, center: 1566, amplitude: 1500, width: 12, color: '#4ecdc4', name: 'Banda 2' },
            { id: 3, center: 1580, amplitude: 3000, width: 18, color: '#45b7d1', name: 'Banda 3' },
            { id: 4, center: 1604, amplitude: 2500, width: 16, color: '#96ceb4', name: 'Banda 4' },
            { id: 5, center: 1620, amplitude: 2000, width: 14, color: '#ffeaa7', name: 'Banda 5' },
            { id: 6, center: 1632, amplitude: 1800, width: 13, color: '#dda0dd', name: 'Banda 6' },
            { id: 7, center: 1667, amplitude: 1500, width: 20, color: '#98d8c8', name: 'Banda 7' }
        ];
        let baseline = 1000;
        let updateTimeout = null;
        let maxIntensity = 10000; // Será ajustado dinamicamente
        let maxBaseline = 15000; // Será ajustado dinamicamente

        // Função para debounce das atualizações
        function debounceUpdate() {
            if (updateTimeout) {
                clearTimeout(updateTimeout);
            }
            updateTimeout = setTimeout(() => {
                updateCalculations();
            }, 50);
        }

        // Função para ajustar limites dos controles dinamicamente
        function adjustControlLimits() {
            if (!spectrumData.length) return;
            
            const maxDataIntensity = Math.max(...spectrumData.map(p => p.intensity));
            
            // Ajustar limites com margem de segurança
            maxIntensity = Math.max(maxDataIntensity * 1.5, 50000); // Mínimo 50k
            maxBaseline = Math.max(maxDataIntensity * 0.5, 15000); // Mínimo 15k
            
            // Atualizar controles de baseline
            const baselineSlider = document.getElementById('baseline');
            const baselineValue = document.getElementById('baselineValue');
            if (baselineSlider) {
                baselineSlider.max = maxBaseline;
                baselineValue.max = maxBaseline;
            }
            
            console.log(`Limites ajustados: Amplitude máxima = ${maxIntensity}, Baseline máximo = ${maxBaseline}`);
        }

        // Função Gaussiana
        function gaussian(x, center, amplitude, width) {
            const sigma = width / (2 * Math.sqrt(2 * Math.log(2)));
            return amplitude * Math.exp(-0.5 * Math.pow((x - center) / sigma, 2));
        }

        // Função para lidar com seleção de arquivo
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                loadFileData(file);
            }
        }

        // Função para carregar dados do arquivo
        async function loadFileData(file) {
            try {
                showLoadingSpinner('Lendo arquivo Excel...');
                
                let jsonData;
                if (file.name.toLowerCase().endsWith('.csv')) {
                    const text = await readFileAsText(file);
                    jsonData = parseCSV(text);
                } else {
                    if (typeof XLSX === 'undefined') {
                        throw new Error('Biblioteca XLSX não carregou. Verifique sua internet ou firewall.');
                    }
                    const arrayBuffer = await readFileAsArrayBuffer(file);
                    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
                    const sheetName = workbook.SheetNames[0];
                    const worksheet = workbook.Sheets[sheetName];
                    jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
                }
                
                console.log('Dados brutos do Excel:', jsonData.slice(0, 5));
                
                // Processar dados
                const processedData = processExcelData(jsonData);
                
                if (processedData.length === 0) {
                    throw new Error('Nenhum dado válido encontrado na região 1500-1700 cm⁻¹');
                }
                
                spectrumData = processedData;
                
                showLoadingSpinner('Ajustando controles para seus dados...');
                
                // Ajustar limites dinamicamente
                adjustControlLimits();
                
                // Ajustar valores iniciais das bandas baseado nos dados
                adjustInitialBandParameters();
                
                await new Promise(resolve => setTimeout(resolve, 500));
                
                initializeInterface();
                
                console.log(`✅ Arquivo carregado com sucesso: ${spectrumData.length} pontos espectrais`);
                console.log(`Intensidade máxima: ${Math.max(...spectrumData.map(p => p.intensity))}`);
                showMessage(`Arquivo carregado com sucesso! ${spectrumData.length} pontos espectrais na região 1500-1700 cm⁻¹`, 'success');
                
            } catch (error) {
                console.error('Erro ao carregar arquivo:', error);
                hideLoadingSpinner();
                showMessage(`Erro ao carregar arquivo: ${error.message}`, 'error');
            }
        }

        // Função para ajustar parâmetros iniciais das bandas
        function adjustInitialBandParameters() {
            if (!spectrumData.length) return;
            
            const maxDataIntensity = Math.max(...spectrumData.map(p => p.intensity));
            const minDataIntensity = Math.min(...spectrumData.map(p => p.intensity));
            
            // Ajustar baseline para um valor mais apropriado
            baseline = Math.min(minDataIntensity * 0.8, maxDataIntensity * 0.1);
            
            // Ajustar amplitudes das bandas proporcionalmente
            const intensityRange = maxDataIntensity - minDataIntensity;
            bands.forEach(band => {
                // Escalar amplitude baseada na intensidade dos dados
                band.amplitude = Math.max(intensityRange * 0.1, band.amplitude);
            });
            
            console.log(`Baseline ajustado para: ${baseline}`);
            console.log(`Amplitudes das bandas ajustadas baseadas na intensidade máxima: ${maxDataIntensity}`);
        }

        // Função para ler arquivo como ArrayBuffer
        function readFileAsArrayBuffer(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    resolve(e.target.result);
                };
                reader.onerror = function() {
                    reject(new Error('Erro ao ler o arquivo'));
                };
                reader.readAsArrayBuffer(file);
            });
        }

        // Função para processar dados do Excel
        function processExcelData(jsonData) {
            console.log(`Processando ${jsonData.length} linhas do Excel`);
            
            // Detectar se primeira linha é cabeçalho
            let startRow = 0;
            if (jsonData.length > 0 && jsonData[0]) {
                const firstCell = jsonData[0][0];
                if (typeof firstCell === 'string' || isNaN(parseFloat(firstCell))) {
                    startRow = 1;
                    console.log('Detectado cabeçalho, iniciando da linha 1');
                }
            }

        // Leitura como texto (para CSV)
        function readFileAsText(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = () => reject(new Error('Erro ao ler o arquivo (texto)'));
                reader.readAsText(file);
            });
        }

        // Parser CSV bem simples (separadores ; ou ,)
        function parseCSV(text) {
            const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
            return lines.map(line => line.split(/\s*[;,]\s*/));
        }

            
            const processedData = [];
            let validCount = 0;
            let rangeCount = 0;
            
            for (let i = startRow; i < jsonData.length; i++) {
                const row = jsonData[i];
                if (row && row.length >= 2) {
                    const wavenumber = parseFloat(row[0]);
                    const intensity = parseFloat(row[1]);
                    
                    if (!isNaN(wavenumber) && !isNaN(intensity)) {
                        validCount++;
                        
                        // Filtrar região de interesse
                        if (wavenumber >= 1500 && wavenumber <= 1700) {
                            processedData.push({
                                wavenumber: wavenumber,
                                intensity: intensity
                            });
                            rangeCount++;
                        }
                    }
                }
            }
            
            console.log(`Dados válidos: ${validCount}, Na região 1500-1700: ${rangeCount}`);
            
            // Ordenar por número de onda
            processedData.sort((a, b) => a.wavenumber - b.wavenumber);
            
            return processedData;
        }

        // Função para carregar dados de exemplo
        function loadExampleData() {
            showLoadingSpinner('Gerando dados de exemplo...');
            
            setTimeout(() => {
                spectrumData = generateExampleData();
                adjustControlLimits();
                adjustInitialBandParameters();
                initializeInterface();
                showMessage('Dados de exemplo carregados! Você pode usar os controles para praticar.', 'success');
            }, 1000);
        }

        // Gerar dados de exemplo (simulando suas intensidades altas)
        function generateExampleData() {
            const data = [];
            for (let i = 1500; i <= 1700; i += 1) {
                // Simular espectro com intensidades similares às suas (~24.000)
                let intensity = 2000 + Math.random() * 1000; // Baseline mais alto
                
                // Adicionar picos grandes nas posições das bandas
                bands.forEach(band => {
                    // Amplificar contribuição para simular suas altas intensidades
                    const contribution = gaussian(i, band.center, band.amplitude * 3.5, band.width);
                    intensity += contribution;
                });
                
                // Pico principal grande em ~1620 (similar ao seu gráfico)
                const mainPeak = gaussian(i, 1620, 15000, 25);
                intensity += mainPeak;
                
                // Adicionar ruído proporcional
                intensity += (Math.random() - 0.5) * 500;
                
                data.push({
                    wavenumber: i,
                    intensity: Math.max(0, intensity)
                });
            }
            return data;
        }

        // Inicializar interface
        function initializeInterface() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('app').style.display = 'block';
            
            setupControls();
            updateCalculations();
            createBandControls();
            updateResultsTable();
            createCharts();
            
            console.log('Interface carregada com sucesso!');
            console.log(`Dados carregados: ${spectrumData.length} pontos espectrais`);
        }

        // Criar gráficos simples
        function createCharts() {
            createMainChart();
            createResidualChart();
        }

        // Criar gráfico principal
        function createMainChart() {
            const container = document.getElementById('mainChart');
            container.innerHTML = '<canvas id="mainCanvas" width="800" height="400"></canvas>';
            
            const canvas = document.getElementById('mainCanvas');
            const ctx = canvas.getContext('2d');
            
            // Configurar canvas para alta resolução
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = rect.height * 2;
            ctx.scale(2, 2);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            drawMainChart(ctx, rect.width, rect.height);
        }

        // Criar gráfico de resíduos
        function createResidualChart() {
            const container = document.getElementById('residualChart');
            container.innerHTML = '<canvas id="residualCanvas" width="800" height="200"></canvas>';
            
            const canvas = document.getElementById('residualCanvas');
            const ctx = canvas.getContext('2d');
            
            // Configurar canvas para alta resolução
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = rect.height * 2;
            ctx.scale(2, 2);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            drawResidualChart(ctx, rect.width, rect.height);
        }

        // Desenhar gráfico principal
        function drawMainChart(ctx, width, height) {
            if (!spectrumData.length) return;
            
            const padding = 60;
            const chartWidth = width - 2 * padding;
            const chartHeight = height - 2 * padding;
            
            // Limpar canvas
            ctx.clearRect(0, 0, width, height);
            
            // Fundo
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);
            
            // Encontrar limites dos dados
            const minX = Math.min(...spectrumData.map(p => p.wavenumber));
            const maxX = Math.max(...spectrumData.map(p => p.wavenumber));
            const minY = 0;
            const maxY = Math.max(...spectrumData.map(p => Math.max(p.intensity, p.fitted || 0)));
            
            // Função para converter coordenadas
            function toCanvas(x, y) {
                const canvasX = padding + ((x - minX) / (maxX - minX)) * chartWidth;
                const canvasY = height - padding - ((y - minY) / (maxY - minY)) * chartHeight;
                return [canvasX, canvasY];
            }
            
            // Desenhar grade
            ctx.strokeStyle = '#f1f5f9';
            ctx.lineWidth = 1;
            ctx.beginPath();
            // Linhas verticais
            for (let i = 0; i <= 10; i++) {
                const x = padding + (i / 10) * chartWidth;
                ctx.moveTo(x, padding);
                ctx.lineTo(x, height - padding);
            }
            // Linhas horizontais
            for (let i = 0; i <= 5; i++) {
                const y = padding + (i / 5) * chartHeight;
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
            }
            ctx.stroke();
            
            // Desenhar eixos
            ctx.strokeStyle = '#d1d5db';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding);
            ctx.stroke();
            
            // Marcações dos eixos
            ctx.fillStyle = '#6b7280';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            
            // Eixo X
            for (let i = 0; i <= 4; i++) {
                const value = minX + (i / 4) * (maxX - minX);
                const [x] = toCanvas(value, 0);
                ctx.fillText(Math.round(value), x, height - padding + 15);
            }
            
            // Eixo Y
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {
                const value = minY + (i / 4) * (maxY - minY);
                const [, y] = toCanvas(0, value);
                ctx.fillText(Math.round(value), padding - 10, y + 3);
            }
            
            // Labels dos eixos
            ctx.fillStyle = '#374151';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Raman Shift (cm⁻¹)', width / 2, height - 15);
            
            ctx.save();
            ctx.translate(20, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Intensidade', 0, 0);
            ctx.restore();
            
            // Desenhar bandas individuais primeiro (se habilitado)
            const showBands = document.getElementById('showBands').checked;
            if (showBands) {
                bands.forEach(band => {
                    // Linha da banda
                    ctx.strokeStyle = band.color;
                    ctx.lineWidth = 3;
                    ctx.globalAlpha = 0.9;
                    ctx.beginPath();
                    
                    let firstPoint = true;
                    spectrumData.forEach((point) => {
                        const bandOnly = gaussian(point.wavenumber, band.center, band.amplitude, band.width);
                        const bandWithBaseline = baseline + bandOnly;
                        const [x, y] = toCanvas(point.wavenumber, bandWithBaseline);
                        
                        if (firstPoint) {
                            ctx.moveTo(x, y);
                            firstPoint = false;
                        } else {
                            ctx.lineTo(x, y);
                        }
                    });
                    ctx.stroke();
                    
                    // Área preenchida da banda
                    ctx.globalAlpha = 0.3;
                    ctx.fillStyle = band.color;
                    ctx.beginPath();
                    
                    // Começar do baseline
                    const [startX, startY] = toCanvas(spectrumData[0].wavenumber, baseline);
                    ctx.moveTo(startX, startY);
                    
                    // Desenhar a curva da banda
                    spectrumData.forEach((point) => {
                        const bandWithBaseline = baseline + gaussian(point.wavenumber, band.center, band.amplitude, band.width);
                        const [x, y] = toCanvas(point.wavenumber, bandWithBaseline);
                        ctx.lineTo(x, y);
                    });
                    
                    // Fechar no baseline
                    const [endX, endY] = toCanvas(spectrumData[spectrumData.length-1].wavenumber, baseline);
                    ctx.lineTo(endX, endY);
                    ctx.closePath();
                    ctx.fill();
                });
                ctx.globalAlpha = 1.0;
            }
            
            // Desenhar baseline
            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            const [baselineStartX, baselineY] = toCanvas(minX, baseline);
            const [baselineEndX] = toCanvas(maxX, baseline);
            ctx.moveTo(baselineStartX, baselineY);
            ctx.lineTo(baselineEndX, baselineY);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Desenhar dados experimentais
            if (spectrumData.length > 0) {
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                spectrumData.forEach((point, i) => {
                    const [x, y] = toCanvas(point.wavenumber, point.intensity);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
                
                // Desenhar ajuste total
                if (spectrumData[0].fitted !== undefined) {
                    ctx.strokeStyle = '#dc2626';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([8, 4]);
                    ctx.beginPath();
                    
                    spectrumData.forEach((point, i) => {
                        const [x, y] = toCanvas(point.wavenumber, point.fitted);
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }
            
            // Legenda
            const legendY = padding + 25;
            const legendX = padding + 20;
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'left';
            
            // Experimental
            ctx.fillStyle = '#000000';
            ctx.fillRect(legendX, legendY - 8, 20, 2);
            ctx.fillText('Experimental', legendX + 30, legendY);
            
            // Ajuste Total
            ctx.fillStyle = '#dc2626';
            ctx.fillRect(legendX + 120, legendY - 8, 15, 2);
            ctx.fillRect(legendX + 140, legendY - 8, 15, 2);
            ctx.fillText('Ajuste Total', legendX + 160, legendY);
            
            // Baseline
            ctx.fillStyle = '#94a3b8';
            for (let i = 0; i < 4; i++) {
                ctx.fillRect(legendX + 260 + i * 8, legendY - 8, 3, 2);
            }
            ctx.fillText('Baseline', legendX + 300, legendY);
            
            // Bandas (se visíveis)
            if (showBands) {
                let legendOffset = 0;
                bands.forEach((band, index) => {
                    if (index < 3) {
                        const yPos = legendY + 20 + legendOffset * 15;
                        ctx.fillStyle = band.color;
                        ctx.fillRect(legendX + index * 100, yPos - 8, 20, 2);
                        ctx.fillText(band.name, legendX + index * 100 + 30, yPos);
                    }
                });
            }
        }

        // Desenhar gráfico de resíduos
        function drawResidualChart(ctx, width, height) {
            if (!spectrumData.length || spectrumData[0].residual === undefined) return;
            
            const padding = 60;
            const chartWidth = width - 2 * padding;
            const chartHeight = height - 2 * padding;
            
            // Limpar canvas
            ctx.clearRect(0, 0, width, height);
            
            // Fundo
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);
            
            // Encontrar limites dos dados
            const minX = Math.min(...spectrumData.map(p => p.wavenumber));
            const maxX = Math.max(...spectrumData.map(p => p.wavenumber));
            const residuals = spectrumData.map(p => p.residual);
            const minY = Math.min(...residuals);
            const maxY = Math.max(...residuals);
            const range = Math.max(Math.abs(minY), Math.abs(maxY));
            const yMin = -range * 1.1;
            const yMax = range * 1.1;
            
            // Função para converter coordenadas
            function toCanvas(x, y) {
                const canvasX = padding + ((x - minX) / (maxX - minX)) * chartWidth;
                const canvasY = height - padding - ((y - yMin) / (yMax - yMin)) * chartHeight;
                return [canvasX, canvasY];
            }
            
            // Desenhar grade
            ctx.strokeStyle = '#f1f5f9';
            ctx.lineWidth = 1;
            ctx.beginPath();
            // Linhas verticais
            for (let i = 0; i <= 10; i++) {
                const x = padding + (i / 10) * chartWidth;
                ctx.moveTo(x, padding);
                ctx.lineTo(x, height - padding);
            }
            // Linhas horizontais
            for (let i = 0; i <= 4; i++) {
                const y = padding + (i / 4) * chartHeight;
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
            }
            ctx.stroke();
            
            // Desenhar eixos
            ctx.strokeStyle = '#d1d5db';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding);
            ctx.stroke();
            
            // Linha zero (mais destacada)
            ctx.strokeStyle = '#6b7280';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 3]);
            const [, zeroY] = toCanvas(minX, 0);
            ctx.beginPath();
            ctx.moveTo(padding, zeroY);
            ctx.lineTo(width - padding, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Marcações dos eixos
            ctx.fillStyle = '#6b7280';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            
            // Eixo X
            for (let i = 0; i <= 4; i++) {
                const value = minX + (i / 4) * (maxX - minX);
                const [x] = toCanvas(value, 0);
                ctx.fillText(Math.round(value), x, height - padding + 15);
            }
            
            // Eixo Y
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {
                const value = yMin + (i / 4) * (yMax - yMin);
                const [, y] = toCanvas(0, value);
                ctx.fillText(Math.round(value), padding - 10, y + 3);
            }
            
            // Labels dos eixos
            ctx.fillStyle = '#374151';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Raman Shift (cm⁻¹)', width / 2, height - 15);
            
            ctx.save();
            ctx.translate(20, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Resíduo', 0, 0);
            ctx.restore();
            
            // Desenhar resíduos
            ctx.strokeStyle = '#6b7280';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            
            spectrumData.forEach((point, i) => {
                const [x, y] = toCanvas(point.wavenumber, point.residual);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // Adicionar pontos para melhor visualização
            ctx.fillStyle = '#6b7280';
            spectrumData.forEach((point, i) => {
                if (i % 5 === 0) {
                    const [x, y] = toCanvas(point.wavenumber, point.residual);
                    ctx.beginPath();
                    ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        }

        // Configurar controles
        function setupControls() {
            const baselineSlider = document.getElementById('baseline');
            const baselineValue = document.getElementById('baselineValue');
            
            // Definir valores baseados nos limites ajustados
            baselineSlider.max = maxBaseline;
            baselineValue.max = maxBaseline;
            baselineSlider.value = baseline;
            baselineValue.value = baseline;
            
            baselineSlider.oninput = function() {
                baseline = parseFloat(this.value);
                baselineValue.value = baseline;
                updateCalculations();
            };
            
            baselineValue.oninput = function() {
                baseline = parseFloat(this.value) || 0;
                baselineSlider.value = baseline;
                updateCalculations();
            };
            
            document.getElementById('showBands').onchange = function() {
                updateCalculations();
            };
        }

        // Criar controles das bandas
        function createBandControls() {
            const container = document.getElementById('bandControls');
            container.innerHTML = '';
            
            bands.forEach(band => {
                const bandDiv = document.createElement('div');
                bandDiv.className = 'band-control';
                bandDiv.style.borderColor = band.color;
                
                // Usar limites ajustados dinamicamente
                const amplitudeMax = Math.max(maxIntensity, 50000); // Garantir mínimo de 50k
                
                bandDiv.innerHTML = `
                    <div class="band-header" style="background-color: ${band.color};">
                        ${band.name}
                    </div>
                    
                    <div class="input-group">
                        <label>Centro: <span id="center-${band.id}-display">${band.center.toFixed(1)}</span> cm⁻¹</label>
                        <input type="range" id="center-${band.id}" min="1500" max="1700" value="${band.center}" step="0.5" 
                               oninput="updateBand(${band.id}, 'center', this.value)">
                    </div>
                    
                    <div class="input-group">
                        <label>Amplitude: <span id="amplitude-${band.id}-display">${Math.round(band.amplitude)}</span> (máx: ${amplitudeMax.toLocaleString()})</label>
                        <input type="range" id="amplitude-${band.id}" min="0" max="${amplitudeMax}" value="${band.amplitude}" step="500"
                               oninput="updateBand(${band.id}, 'amplitude', this.value)">
                        <input type="number" id="amplitude-${band.id}-number" min="0" max="${amplitudeMax}" value="${band.amplitude}" step="100"
                               oninput="updateBandFromNumber(${band.id}, 'amplitude', this.value)" style="margin-top: 5px;">
                    </div>
                    
                    <div class="input-group">
                        <label>FWHM: <span id="width-${band.id}-display">${band.width.toFixed(1)}</span></label>
                        <input type="range" id="width-${band.id}" min="5" max="80" value="${band.width}" step="0.5"
                               oninput="updateBand(${band.id}, 'width', this.value)">
                    </div>
                `;
                
                container.appendChild(bandDiv);
            });
        }

        // Atualizar parâmetros da banda via slider
        function updateBand(id, property, value) {
            const band = bands.find(b => b.id === id);
            if (band) {
                band[property] = parseFloat(value);
                document.getElementById(`${property}-${id}-display`).textContent = 
                    property === 'amplitude' ? Math.round(band[property]) : band[property].toFixed(1);
                
                // Sincronizar com input numérico se for amplitude
                if (property === 'amplitude') {
                    const numberInput = document.getElementById(`amplitude-${id}-number`);
                    if (numberInput) numberInput.value = band[property];
                }
                
                updateCalculations();
            }
        }

        // Atualizar parâmetros da banda via input numérico
        function updateBandFromNumber(id, property, value) {
            const band = bands.find(b => b.id === id);
            if (band) {
                const numValue = parseFloat(value) || 0;
                band[property] = numValue;
                
                // Sincronizar com slider
                const slider = document.getElementById(`${property}-${id}`);
                if (slider) {
                    slider.value = numValue;
                    // Ajustar max do slider se necessário
                    if (numValue > parseFloat(slider.max)) {
                        slider.max = numValue * 1.2;
                    }
                }
                
                document.getElementById(`${property}-${id}-display`).textContent = 
                    property === 'amplitude' ? Math.round(band[property]) : band[property].toFixed(1);
                
                updateCalculations();
            }
        }

        // Atualizar cálculos
        function updateCalculations() {
            if (!spectrumData.length) return;
            
            // Calcular ajuste para cada ponto
            let totalSumSquares = 0;
            let residualSumSquares = 0;
            let totalIntensity = 0;
            
            spectrumData.forEach(point => {
                totalIntensity += point.intensity;
            });
            
            const meanIntensity = totalIntensity / spectrumData.length;
            
            spectrumData.forEach(point => {
                // Calcular contribuição total começando com baseline
                let fitted = baseline;
                
                // Adicionar contribuição de cada banda e armazenar individualmente
                bands.forEach(band => {
                    const bandContribution = gaussian(point.wavenumber, band.center, band.amplitude, band.width);
                    fitted += bandContribution;
                    // Armazenar contribuição individual da banda
                    point[`band_${band.id}`] = bandContribution;
                });
                
                point.fitted = fitted;
                point.residual = point.intensity - fitted;
                
                totalSumSquares += Math.pow(point.intensity - meanIntensity, 2);
                residualSumSquares += Math.pow(point.residual, 2);
            });
            
            // Calcular estatísticas
            const rSquared = totalSumSquares > 0 ? 1 - (residualSumSquares / totalSumSquares) : 0;
            const rmse = Math.sqrt(residualSumSquares / spectrumData.length);
            const mae = spectrumData.reduce((sum, p) => sum + Math.abs(p.residual), 0) / spectrumData.length;
            
            // Atualizar interface
            document.getElementById('rSquared').textContent = rSquared.toFixed(4);
            document.getElementById('rmse').textContent = rmse.toFixed(2);
            document.getElementById('mae').textContent = mae.toFixed(2);
            document.getElementById('dataPoints').textContent = spectrumData.length;
            
            updateResultsTable();
            
            // Atualizar gráficos se existirem
            if (document.getElementById('mainCanvas')) {
                createCharts();
            }
            
            console.log('Cálculos atualizados - R²:', rSquared.toFixed(4), 'RMSE:', rmse.toFixed(2));
        }

        // Atualizar tabela de resultados
        function updateResultsTable() {
            const tbody = document.getElementById('resultsTableBody');
            tbody.innerHTML = '';
            
            bands.forEach(band => {
                const area = band.amplitude * band.width * Math.sqrt(2 * Math.PI) / (2 * Math.sqrt(2 * Math.log(2)));
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td style="color: ${band.color}; font-weight: bold;">${band.name}</td>
                    <td>${band.center.toFixed(1)}</td>
                    <td>${Math.round(band.amplitude).toLocaleString()}</td>
                    <td>${band.width.toFixed(1)}</td>
                    <td>${Math.round(area).toLocaleString()}</td>
                `;
                tbody.appendChild(row);
            });
        }

        // Modal de exportação
        function showExportModal() {
            document.getElementById('exportModal').classList.add('show');
        }

        function hideExportModal() {
            document.getElementById('exportModal').classList.remove('show');
        }

        // Exportar dados
        function performExport() {
            const format = document.getElementById('exportFormat').value;
            const includeBands = document.getElementById('includeBands').checked;
            const includeSpectrum = document.getElementById('includeSpectrum').checked;
            const includeStats = document.getElementById('includeStats').checked;
            
            const data = {};
            
            if (includeBands) {
                data.bands = bands.map(band => ({
                    name: band.name,
                    center: band.center,
                    amplitude: band.amplitude,
                    width: band.width,
                    area: band.amplitude * band.width * Math.sqrt(2 * Math.PI) / (2 * Math.sqrt(2 * Math.log(2))),
                    color: band.color
                }));
            }
            
            if (includeSpectrum) {
                data.spectrum = spectrumData.map(point => ({
                    wavenumber: point.wavenumber,
                    experimental: point.intensity,
                    fitted: point.fitted,
                    residual: point.residual
                }));
            }
            
            if (includeStats) {
                data.statistics = {
                    rSquared: parseFloat(document.getElementById('rSquared').textContent),
                    rmse: parseFloat(document.getElementById('rmse').textContent),
                    mae: parseFloat(document.getElementById('mae').textContent),
                    baseline: baseline,
                    dataPoints: spectrumData.length,
                    maxIntensity: Math.max(...spectrumData.map(p => p.intensity)),
                    analysisDate: new Date().toISOString()
                };
            }
            
            // Criar arquivo para download
            let content, filename, mimeType;
            
            if (format === 'json') {
                content = JSON.stringify(data, null, 2);
                filename = `raman_deconvolution_${new Date().toISOString().split('T')[0]}.json`;
                mimeType = 'application/json';
            } else if (format === 'csv') {
                content = convertToCSV(data);
                filename = `raman_deconvolution_${new Date().toISOString().split('T')[0]}.csv`;
                mimeType = 'text/csv';
            } else {
                content = convertToText(data);
                filename = `raman_deconvolution_${new Date().toISOString().split('T')[0]}.txt`;
                mimeType = 'text/plain';
            }
            
            // Download
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
            
            hideExportModal();
            
            // Mostrar mensagem de sucesso
            showMessage(`Arquivo ${filename} exportado com sucesso!`, 'success');
        }

        // Converter para CSV
        function convertToCSV(data) {
            let csv = '';
            
            if (data.spectrum) {
                csv += 'Raman Shift (cm-1),Experimental,Fitted,Residual\\n';
                data.spectrum.forEach(point => {
                    csv += `${point.wavenumber},${point.experimental},${point.fitted},${point.residual}\\n`;
                });
            }
            
            return csv;
        }

        // Converter para texto
        function convertToText(data) {
            let text = 'RESULTADOS DA DECONVOLUÇÃO RAMAN\\n';
            text += '=====================================\\n\\n';
            
            if (data.statistics) {
                text += 'ESTATÍSTICAS DE AJUSTE:\\n';
                text += `R² = ${data.statistics.rSquared}\\n`;
                text += `RMSE = ${data.statistics.rmse}\\n`;
                text += `MAE = ${data.statistics.mae}\\n`;
                text += `Baseline = ${data.statistics.baseline}\\n`;
                text += `Pontos de dados = ${data.statistics.dataPoints}\\n`;
                text += `Intensidade máxima = ${data.statistics.maxIntensity}\\n\\n`;
            }
            
            if (data.bands) {
                text += 'PARÂMETROS DAS BANDAS:\\n';
                data.bands.forEach(band => {
                    text += `${band.name}: Centro=${band.center} cm⁻¹, Amplitude=${band.amplitude}, FWHM=${band.width}, Área=${Math.round(band.area)}\\n`;
                });
                text += '\\n';
            }
            
            return text;
        }

        // Mostrar mensagem
        function showMessage(message, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = type === 'success' ? 'success-message' : 'error-message';
            messageDiv.innerHTML = `
                <div style="display: flex; justify-content: between; align-items: center;">
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; font-size: 18px; cursor: pointer; margin-left: 10px;">&times;</button>
                </div>
            `;
            
            document.querySelector('.container').insertBefore(messageDiv, document.querySelector('#loading'));
            
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.remove();
                }
            }, 5000);
        }

        // Configurar drag and drop
        function setupDragAndDrop() {
            const dropZone = document.querySelector('#dropzone');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });

            function highlight(e) {
                dropZone.style.backgroundColor = '#f0f9ff';
                dropZone.style.borderColor = '#3b82f6';
            }

            function unhighlight(e) {
                dropZone.style.backgroundColor = '';
                dropZone.style.borderColor = '';
            }

            dropZone.addEventListener('drop', handleDrop, false);

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;

                if (files.length > 0) {
                    const file = files[0];
                    if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
                        loadFileData(file);
                    } else {
                        showMessage('Por favor, selecione um arquivo Excel (.xlsx ou .xls)', 'error');
                    }
                }
            }
        }

        // Inicializar aplicação
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🔬 Aplicação Raman carregada - Interface de Upload ativa');
            console.log('📈 Controles ajustados para altas intensidades');
            setupDragAndDrop();
        });
    </script>
</body>
</html>
        '''

    def start_server(self):
        """Inicia o servidor Flask"""
        try:
            self.server = make_server('localhost', self.port, self.app, threaded=True)
            print(f"🔬 Servidor Raman iniciado em http://localhost:{self.port}")
            print("🌐 Abrindo navegador...")
            
            # Abrir navegador após pequeno delay
            threading.Timer(2.0, lambda: webbrowser.open(f'http://localhost:{self.port}')).start()
            
            # Iniciar servidor
            self.server.serve_forever()
            
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"⚠ Porta {self.port} já está em uso. Tentando porta alternativa...")
                self.port += 1
                if self.port < 8090:  # Tentar até 8090
                    self.start_server()
                else:
                    print("⚠ Não foi possível encontrar uma porta disponível.")
            else:
                print(f"⚠ Erro ao iniciar servidor: {e}")
    
    def stop_server(self):
        """Para o servidor"""
        if self.server:
            self.server.shutdown()
            print("🛑 Servidor parado.")

def check_requirements():
    """Verifica se as dependências estão instaladas"""
    required_packages = ['flask', 'werkzeug']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠ Pacotes Python necessários não encontrados:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Instale com: pip install " + " ".join(missing_packages))
        return False
    return True

def check_data_file():
    """Verifica se o arquivo Raman.xlsx existe"""
    if not os.path.exists('Raman.xlsx'):
        print("⚠️  Arquivo 'Raman.xlsx' não encontrado.")
        print("💡 A aplicação funcionará com dados de exemplo.")
        print("   Para usar seus dados, coloque o arquivo 'Raman.xlsx' na pasta atual.")
        return True  # Continuar mesmo sem o arquivo
    print("✅ Arquivo Raman.xlsx encontrado.")
    return True

def main():
    """Função principal"""
    print("=" * 60)
    print("🔬 DECONVOLUÇÃO ESPECTRAL RAMAN")
    print("   Versão com Controles Ajustados para Altas Intensidades")
    print("=" * 60)
    
    # Verificar dependências
    if not check_requirements():
        return
    
    # Verificar arquivo de dados
    check_data_file()
    
    print("✅ Iniciando aplicação...")
    print("📈 NOVO: Controles ajustados automaticamente baseados nos dados")
    print("   - Amplitude: até 50.000+ (ajustado dinamicamente)")
    print("   - Baseline: até 15.000+ (ajustado dinamicamente)")
    print("   - Inputs numéricos para valores precisos")
    
    # Criar e iniciar aplicação
    app = RamanApp()
    
    try:
        app.start_server()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrompido pelo usuário.")
        app.stop_server()
    except Exception as e:
        print(f"\n⚠ Erro inesperado: {e}")
        app.stop_server()

if __name__ == "__main__":
    main()
