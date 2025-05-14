import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np
from flask import Response, send_file
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
# Modelos de ML
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from estatistica import gerar_dados_estatisticos
from flask import send_from_directory

# from optimize_models import objective, executar_otimizacao

app = Flask(__name__)
app.secret_key = 'minha_chave_secreta_123' 

def treinar_e_salvar_modelos():
    dados_treino = pd.read_csv('C:/Users/andreJoas/Desktop/basetratada.csv')
    dados_teste = pd.read_csv('C:/Users/andreJoas/Desktop/basetest.csv')

    variaveis_alvo = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

    X_treino = dados_treino.drop(columns=variaveis_alvo)
    y_treino = dados_treino[variaveis_alvo]

    modelos = {
        'HistGradientBoosting': HistGradientBoostingClassifier(
            learning_rate = 0.08617528911440171,
max_iter = 122,
max_leaf_nodes = 25,
max_depth = 7,
min_samples_leaf = 14,
l2_regularization = 1.2021887775383737
        ),
        'LGBMClassifier': LGBMClassifier(
           learning_rate = 0.05372546253527757,
n_estimators = 274,
max_depth = 5,
num_leaves = 75,
subsample = 0.6679386886385299,
colsample_bytree = 0.790719895811263

        ),
        'RandomForest': RandomForestClassifier(
            n_estimators = 194,
max_depth = 12,
min_samples_split = 5,
min_samples_leaf = 2,
max_features = None

        ),
        'GradientBoosting': GradientBoostingClassifier(
            learning_rate = 0.07394557890599904,
n_estimators = 217,
max_depth = 5,
min_samples_split = 10,
min_samples_leaf = 2,
subsample = 0.7238129722616937

        ),
        'LogisticRegression': LogisticRegression(
            C = 7.568891261692385,
penalty = "l2",
solver = "liblinear",
max_iter = 388
        ),
        'XGBClassifier': XGBClassifier(
            n_estimators = 156,
max_depth = 10,
learning_rate = 0.09402527458200952,
subsample = 0.9101557784406703,
colsample_bytree = 0.7311692223011678,
gamma = 0.12021779769124034,

        )
    }

    for modelo_nome, modelo in modelos.items():
        print(f"\n🔍 Treinando modelo: {modelo_nome}")
        df_modelo_resultado = pd.DataFrame(index=dados_teste.index)

        for falha in variaveis_alvo:
            print(f"  → Treinando para: {falha}")
            modelo.fit(X_treino, y_treino[falha])
            probas = modelo.predict_proba(dados_teste)[:, 1]
            df_modelo_resultado[falha] = probas
            print(f"    📊 Primeiras previsões para {falha}:\n{df_modelo_resultado[falha].head().to_string(index=False, float_format='%.4f')}")

        nome_arquivo = f"previsoes_{modelo_nome}.csv"
        df_modelo_resultado.to_csv(nome_arquivo, index=False)
        print(f"✅ Resultados salvos em: {nome_arquivo}")

@app.route('/detalhes')
def detalhes():
    arquivos_csv = [arq for arq in os.listdir() if arq.startswith('previsoes_') and arq.endswith('.csv')]

    tabelas_html = ""
    for arq in arquivos_csv:
        df = pd.read_csv(arq)

        medias = df.mean() * 100  # Média percentual por coluna
        medias_html = "<ul>"
        for coluna, media in medias.items():
            medias_html += f"<li><strong>{coluna}</strong>: {media:.2f}%</li>"
        medias_html += "</ul>"

        nome_modelo = arq.replace("previsoes_", "").replace(".csv", "")
        tabela_formatada = df.to_html(classes='table table-striped table-bordered custom-table', index=False)

        tabelas_html += f"""
        <div class="tabela-bloco modelo-section">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <h3>🧠 Modelo: {nome_modelo}</h3>
                <a href="/download_csv/{arq}" class="btn btn-success btn-sm">⬇ Baixar CSV</a>
            </div>
            <p><strong>Média das previsões por classe:</strong>{medias_html}</p>
            <div class="scroll-tabela">{tabela_formatada}</div>
        </div>
        """

    return render_template("detalhes.html", tabelas_html=tabelas_html)


@app.route('/download_csv/<path:filename>')
def download_csv(filename):
    return send_from_directory(directory='.', path=filename, as_attachment=True)


@app.route('/')
def home():
    graficos_html, tabela_html = gerar_dados_estatisticos()
    return render_template("index.html", graficos_html=graficos_html, tabela_html=tabela_html)

def executar_pipeline():
    from processamento.pipeline import pipeline_completo, normalizar_dados, tratar_categoricas

    caminho_treino = 'base/bootcamp_train.csv'
    caminho_teste = 'base/bootcamp_test.csv'

    df_train = pd.read_csv(caminho_treino)
    df_test = pd.read_csv(caminho_teste)

    colunas_categoricas = [
        'tipo_do_aço_A300', 'tipo_do_aço_A400',
        'falha_1', 'falha_2', 'falha_3',
        'falha_4', 'falha_5', 'falha_6',
        'falha_outros'
    ]

    df_train_tratado, df_test_tratado = pipeline_completo(df_train, df_test, colunas_categoricas)
    df_train_final = tratar_categoricas(normalizar_dados(df_train_tratado))
    df_test_final = tratar_categoricas(normalizar_dados(df_test_tratado))

    df_train_final.to_csv('C:/Users/andreJoas/Desktop/basetratada.csv', index=False)
    df_test_final.to_csv('C:/Users/andreJoas/Desktop/basetest.csv', index=False)
    print("✅ Bases tratadas e exportadas.")

# ---------
def treinar_e_salvar_modelos_com_parametros(params):
    dados_treino = pd.read_csv('C:/Users/andreJoas/Desktop/basetratada.csv')
    dados_teste = pd.read_csv('C:/Users/andreJoas/Desktop/basetest.csv')

    variaveis_alvo = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
    X_treino = dados_treino.drop(columns=variaveis_alvo)
    y_treino = dados_treino[variaveis_alvo]

    # Modelos com os parâmetros passados
    modelos = {
        'HistGradientBoosting': HistGradientBoostingClassifier(**params),
        'LGBMClassifier': LGBMClassifier(**params),
        'RandomForest': RandomForestClassifier(**params),
        'GradientBoosting': GradientBoostingClassifier(**params),
        'LogisticRegression': LogisticRegression(**params),
        'XGBClassifier': XGBClassifier(**params),
    }

    for modelo_nome, modelo in modelos.items():
        print(f"\n🔍 Treinando modelo: {modelo_nome}")
        df_modelo_resultado = pd.DataFrame(index=dados_teste.index)

        for falha in variaveis_alvo:
            print(f"  → Treinando para: {falha}")
            modelo.fit(X_treino, y_treino[falha])
            probas = modelo.predict_proba(dados_teste)[:, 1]
            df_modelo_resultado[falha] = probas
            print(f"    📊 Primeiras previsões para {falha}:\n{df_modelo_resultado[falha].head().to_string(index=False, float_format='%.4f')}")

        nome_arquivo = f"previsoes_{modelo_nome}.csv"
        df_modelo_resultado.to_csv(nome_arquivo, index=False)
        print(f"✅ Resultados salvos em: {nome_arquivo}")



@app.route('/treinar_modelo_form', methods=['GET'])
def treinar_modelo_form():
    return render_template('otimizar_modelo.html')



def objective(trial, modelo_nome, X, y_multioutput):
    if modelo_nome == 'HistGradientBoosting':
        params = {
            'loss': 'log_loss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_iter': trial.suggest_int('max_iter', 100, 300),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 2.0),
        }
        model_cls = HistGradientBoostingClassifier
    elif modelo_nome == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        model_cls = RandomForestClassifier
    elif modelo_nome == 'GradientBoosting':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        }
        model_cls = GradientBoostingClassifier
    elif modelo_nome == 'LogisticRegression':
        params = {
            'C': trial.suggest_float('C', 0.01, 10),
            'penalty': trial.suggest_categorical('penalty', ['l2']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
        }
        model_cls = LogisticRegression
    elif modelo_nome == 'XGBClassifier':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
        model_cls = XGBClassifier
        params.update({'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'})
    elif modelo_nome == 'LGBMClassifier':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        model_cls = LGBMClassifier
    else:
        raise ValueError("Modelo não suportado.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []

    for coluna in y_multioutput.columns:
        y_col = y_multioutput[coluna]
        model = model_cls(**params)
        score = cross_val_score(model, X, y_col, scoring='f1', cv=cv, n_jobs=-1)
        f1s.append(np.mean(score))

    return np.mean(f1s)


# Ensure to include this function
def optimize_model(modelo_nome, X, y_multioutput):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, modelo_nome, X, y_multioutput), n_trials=2, timeout=1800)
    return study.best_params

# Rota para otimizar o modelo
@app.route('/otimizar_modelo', methods=['GET', 'POST'])
def otimizar_modelo():
    try:
        # Carregar os dados
        df_treino = pd.read_csv('basetratada/basetratada.csv')
        variaveis_alvo = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
        X = df_treino.drop(columns=variaveis_alvo)
        y_multioutput = df_treino[variaveis_alvo]

        if request.method == 'POST':
            modelo_nome = request.form.get('modelo_nome')  # Pega o nome do modelo do formulário
            print(f"Modelo selecionado: {modelo_nome}")  # Verificar se o valor do modelo está sendo capturado
            
            # Rodar a otimização com os parâmetros do Optuna
            melhores_parametros = optimize_model(modelo_nome, X, y_multioutput)
            print(f"Melhores parâmetros: {melhores_parametros}")  # Verificar se a função está retornando os parâmetros
            
            # Exibir os melhores parâmetros no HTML
            flash(f"Melhores parâmetros encontrados para {modelo_nome}: {melhores_parametros}", "success")
            
            # Redireciona para a página de resultados, passando os parâmetros
            print("Otimização concluída, redirecionando para os resultados.")
            return render_template('otimizacao_resultados.html', 
                                   melhores_parametros=melhores_parametros, 
                                   modelo_nome=modelo_nome)  # Passando 'modelo_nome' para o template
        
        return render_template('otimizar_modelo.html')
    
    except Exception as e:
        print(f"Erro: {e}")  # Verificar qualquer erro ocorrido
        flash(f"Erro ao otimizar o modelo: {e}", "danger")
        return redirect(url_for('home'))  # Caso haja erro, redireciona para a home

@app.route('/baixar_parametros/<formato>')
def baixar_parametros(formato):
    # Recuperar os melhores parâmetros da sessão ou de uma variável global
    melhores_parametros = request.args.get('melhores_parametros', None)
    print(f"Melhores parâmetros recebidos: {melhores_parametros}")  # Verificar se os parâmetros estão sendo passados corretamente
    
    if not melhores_parametros:
        flash("Não foi possível encontrar os parâmetros para download", "danger")
        return redirect(url_for('otimizar_modelo'))

    melhores_parametros = json.loads(melhores_parametros)  # Garantir que seja um dicionário

    # Caso o formato seja JSON
    if formato == 'json':
        return Response(
            json.dumps(melhores_parametros, indent=4),
            mimetype='application/json',
            headers={"Content-Disposition": "attachment;filename=parametros.json"}
        )
    
    # Se o formato não for válido, redireciona de volta
    return redirect(url_for('otimizacao_de_resultados'))

@app.route('/otimizacao_de_resultados')
def otimizacao_de_resultados():
    # Renderiza a página de resultados da otimização
    return render_template('otimizacao_resultados.html')

@app.route('/treinar_modelo', methods=['POST'])
def treinar_modelo_com_parametros():
    try:
        # Pegando os parâmetros do formulário
        params = {
            'learning_rate': float(request.form.get('learning_rate')),
            'max_iter': int(request.form.get('max_iter')),
            'max_depth': int(request.form.get('max_depth')),
            'min_samples_leaf': int(request.form.get('min_samples_leaf')),
            'n_estimators': int(request.form.get('n_estimators'))
        }
        
        # Passando os parâmetros para a função de treinamento
        treinar_e_salvar_modelos_com_parametros(params)
        
        flash("Modelo treinado com sucesso!", "success")
        return redirect(url_for('detalhes'))
    
    except Exception as e:
        flash(f"Erro ao treinar o modelo: {e}", "danger")
        return redirect(url_for('home'))


if __name__ == '__main__':
    arquivos_previsoes = [f for f in os.listdir() if f.startswith('previsoes_') and f.endswith('.csv')]

    if not arquivos_previsoes:
        print("🔁 Nenhum arquivo de previsão encontrado. Iniciando pipeline e treinamento...")
        executar_pipeline()
        treinar_e_salvar_modelos()
        gerar_dados_estatisticos()
        print("✅ Pipeline concluído.")
    else:
        print("✅ Arquivos de previsão encontrados. Pulando treinamento e processamento.")

    print("🚀 Iniciando servidor Flask...")
 
    app.run(debug=True)

