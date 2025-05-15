import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Esta função reúne os dados de previsão de diferentes modelos salvos em arquivos CSV,
# calcula estatísticas como média, desvio padrão e entropia para cada tipo de falha,
# gera gráficos de distribuição usando Plotly e monta uma tabela HTML com as estatísticas.
# Também inclui um botão para download da tabela em formato CSV, facilitando a análise e comparação dos modelos.


def gerar_dados_estatisticos():

    arquivos_modelos = {f.split('_')[1].split('.')[0]: f 
                        for f in os.listdir() if f.startswith('previsoes_') and f.endswith('.csv')}
    
    variaveis_alvo = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
    df_resultado = pd.DataFrame()

   
    for modelo_nome, arquivo in arquivos_modelos.items():
        df = pd.read_csv(arquivo)
        for falha in variaveis_alvo:
            coluna_nome = f'{falha}_{modelo_nome}'
            df_resultado[coluna_nome] = df[falha]

    graficos_html = '<div style="max-height: 600px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">'
    for falha in variaveis_alvo:
        fig = go.Figure()
        for modelo_nome in arquivos_modelos:
            coluna = f'{falha}_{modelo_nome}'
            fig.add_trace(go.Histogram(
                x=df_resultado[coluna],
                name=modelo_nome,
                opacity=0.5,
                histnorm="probability density"
            ))
        fig.update_layout(
            title=f'Distribuição: {falha}',
            xaxis_title='Probabilidade',
            yaxis_title='Densidade',
            barmode='overlay',
            height=400
        )
       
        graficos_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    graficos_html += '</div>'

    # Estatísticas
    estatisticas = []
    for modelo_nome in arquivos_modelos:
        for falha in variaveis_alvo:
            coluna = f'{falha}_{modelo_nome}'
            probs = df_resultado[coluna]
            entropia = -np.mean(probs * np.log2(probs + 1e-9) + (1 - probs) * np.log2(1 - probs + 1e-9))
            estatisticas.append({
                'Modelo': modelo_nome,
                'Falha': falha,
                'Média (%)': probs.mean() * 100,
                'Desvio Padrão': probs.std(),
                'Entropia Média': entropia
            })

    df_estatisticas = pd.DataFrame(estatisticas)
    df_estatisticas[['Média (%)', 'Desvio Padrão', 'Entropia Média']] = df_estatisticas[[
    'Média (%)', 'Desvio Padrão', 'Entropia Média'
    ]].round(2)

    # Converter para CSV
    csv_data = df_estatisticas.to_csv(index=False)

    tabela_html = df_estatisticas.to_html(index=False, classes="table table-striped custom-table", border=0)
    tabela_scroll = f'''
        <div style="max-height: 300px; overflow:auto; margin-top: 20px; border: 1px solid #ccc; padding: 10px;">
            {tabela_html}
        </div>
        <button id="download-btn" class="btn btn-success mt-3">🔽 Baixar Tabela CSV</button>

        <script>
        document.getElementById("download-btn").addEventListener("click", function() {{
            var csvContent = `{csv_data}`;
            var blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
            var link = document.createElement('a');
            if (link.download !== undefined) {{
                var url = URL.createObjectURL(blob);
                link.setAttribute('href', url);
                link.setAttribute('download', 'estatisticas_modelos.csv');
                link.click();
            }}
        }});
        </script>
    '''

    return graficos_html, tabela_scroll
