# import optuna
# from xgboost import XGBClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# import pandas as pd
# import numpy as np
# import json

# # Carregar os dados
# df_treino = pd.read_csv('C:/Users/andreJoas/Desktop/PROJETO_FINAL/basetratada/basetratada.csv')
# variaveis_alvo = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

# X = df_treino.drop(columns=variaveis_alvo)
# y_multioutput = df_treino[variaveis_alvo]

# # Função objetivo para Optuna
# def objective(trial, modelo_nome, X, y_multioutput):
#     if modelo_nome == 'HistGradientBoosting':
#         params = {
#             'loss': 'log_loss',
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'max_iter': trial.suggest_int('max_iter', 100, 300),
#             'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
#             'max_depth': trial.suggest_int('max_depth', 3, 15),
#             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
#             'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 2.0),
#         }
#         model_cls = HistGradientBoostingClassifier
#     elif modelo_nome == 'RandomForest':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'max_depth': trial.suggest_int('max_depth', 5, 15),
#             'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#             'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#         }
#         model_cls = RandomForestClassifier
#     elif modelo_nome == 'GradientBoosting':
#         params = {
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'max_depth': trial.suggest_int('max_depth', 3, 15),
#             'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         }
#         model_cls = GradientBoostingClassifier
#     elif modelo_nome == 'LogisticRegression':
#         params = {
#             'C': trial.suggest_float('C', 0.01, 10),
#             'penalty': trial.suggest_categorical('penalty', ['l2']),
#             'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
#             'max_iter': trial.suggest_int('max_iter', 100, 500),
#         }
#         model_cls = LogisticRegression
#     elif modelo_nome == 'XGBClassifier':
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'max_depth': trial.suggest_int('max_depth', 3, 15),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'gamma': trial.suggest_float('gamma', 0.0, 5.0),
#         }
#         model_cls = XGBClassifier
#         params.update({'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'})
#     elif modelo_nome == 'LGBMClassifier':
#         params = {
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#             'max_depth': trial.suggest_int('max_depth', 3, 15),
#             'num_leaves': trial.suggest_int('num_leaves', 31, 100),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         }
#         model_cls = LGBMClassifier
#     else:
#         raise ValueError("Modelo não suportado.")

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     f1s = []

#     for coluna in y_multioutput.columns:
#         y_col = y_multioutput[coluna]
#         model = model_cls(**params)
#         score = cross_val_score(model, X, y_col, scoring='f1', cv=cv, n_jobs=-1)
#         f1s.append(np.mean(score))

#     return np.mean(f1s)


# def optimize_model(modelo_nome, X, y_multioutput):
#     study = optuna.create_study(direction='maximize')
#     study.optimize(lambda trial: objective(trial, modelo_nome, X, y_multioutput), n_trials=30, timeout=1800)
#     return study.best_params



# def executar_otimizacao():
#     resultados = {}
#     print("🔍 Iniciando otimização para o conjunto de todas as variáveis alvo...")

#     modelos = ['HistGradientBoosting', 'RandomForest', 'GradientBoosting', 'LogisticRegression', 'XGBClassifier', 'LGBMClassifier']

#     for modelo_nome in modelos:
#         print(f"  🔄 Otimizando: {modelo_nome}...")
#         melhores_parametros = optimize_model(modelo_nome, X, y_multioutput)
#         resultados[modelo_nome] = melhores_parametros

#     # Salvar os resultados
#     with open('melhores_parametros_conjunto_variaveis.json', 'w') as f:
#         json.dump(resultados, f, indent=4)

#     print("\n✅ Resultados salvos em 'melhores_parametros_conjunto_variaveis.json'.")

