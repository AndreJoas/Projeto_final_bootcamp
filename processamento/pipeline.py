import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



'''''
                                 Pipeline completo de pré-processamento de dados para treino e teste
                                 Este conjunto de funções realiza as seguintes etapas principais:

1. Tratamento de valores inválidos (negativos) em colunas numéricas, substituindo-os pela mediana da coluna.
2. Identificação e tratamento de outliers usando o método do IQR (Intervalo Interquartil),
     substituindo outliers por NaN e depois preenchendo com a mediana do conjunto de treino.
3. Padronização de categorias textuais para valores consistentes como "Sim" e "Não",
    facilitando a manipulação e interpretação das variáveis categóricas.
4. Conversão de valores booleanos e strings relacionadas (ex: 'true', 'false') para "sim" e "não",
    garantindo uniformidade nos dados categóricos.
5. Aplicação das transformações em ambos os datasets de treino e teste,
    para manter coerência entre os conjuntos.
6. Tratamento específico para as colunas 'tipo_do_aço_A300' e 'tipo_do_aço_A400',
    aplicando regras lógicas para preencher valores faltantes em 'tipo_do_aço_A400' baseado em 'tipo_do_aço_A300'.
7. Normalização dos dados numéricos usando MinMaxScaler,
    escalando os valores para o intervalo [0,1], o que ajuda a evitar vieses em modelos sensíveis à escala.
8. Codificação das colunas categóricas (texto) em valores numéricos utilizando LabelEncoder,
   facilitando a entrada dos dados em modelos de machine learning.

                                        O pipeline garante que as transformações sejam consistentes entre treino e teste,
                                        evitando vazamento de dados e melhorando a qualidade do modelo final.
'''
def tratar_valores_invalidos(df, coluna):
    df[coluna] = df[coluna].apply(lambda x: np.nan if x < 0 else x)
    mediana = df[coluna].median()
    df[coluna] = df[coluna].fillna(mediana)
    return df, mediana

def tratar_outliers_com_IQR(df, coluna, mediana_treino):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)
    df[coluna] = df[coluna].mask(outliers, np.nan)
    df[coluna] = df[coluna].fillna(mediana_treino)
    return df

def padronizar_categorias(valor):
    if pd.isnull(valor):
        return pd.NA
    valor = str(valor).strip().lower()
    if valor in ['sim', 's', '1', 'true', 'yes', 'y', 'verdadeiro']:
        return 'Sim'
    elif valor in ['não', 'nao', 'n', '0', 'false', '-', 'na', 'nan', 'falso']:
        return 'Não'
    else:
        return valor

def tratar_valor_booleano(valor):
    if isinstance(valor, (bool, np.bool_)):
        return 'sim' if valor else 'não'
    if isinstance(valor, str):
        valor_lower = valor.strip().lower()
        if valor_lower in ['true', '1', 'verdadeiro']:
            return 'sim'
        elif valor_lower in ['false', '0', 'falso']:
            return 'não'
        return valor_lower
    return valor

def pipeline_completo(df_train, df_test, colunas_categoricas=[], colunas_excecao=[]):
    df_train_proc = df_train.copy()
    df_test_proc = df_test.copy()

    for df in [df_train_proc, df_test_proc]:
        for coluna in colunas_categoricas:
            if coluna in df.columns:
                df[coluna] = df[coluna].apply(tratar_valor_booleano)

    colunas_numericas = df_train.select_dtypes(include=np.number).columns.tolist()

    for coluna in colunas_numericas:
        if coluna not in colunas_excecao:
            df_train_proc, mediana = tratar_valores_invalidos(df_train_proc, coluna)
            if coluna in df_test_proc.columns:
                df_test_proc[coluna] = df_test_proc[coluna].apply(lambda x: np.nan if x < 0 else x).fillna(mediana)

        mediana_treino = df_train_proc[coluna].median()
        df_train_proc = tratar_outliers_com_IQR(df_train_proc, coluna, mediana_treino)
        if coluna in df_test_proc.columns:
            df_test_proc = tratar_outliers_com_IQR(df_test_proc, coluna, mediana_treino)

    for coluna in colunas_categoricas:
        if coluna in df_train_proc.columns:
            df_train_proc[coluna] = df_train_proc[coluna].apply(padronizar_categorias)
        if coluna in df_test_proc.columns:
            df_test_proc[coluna] = df_test_proc[coluna].apply(padronizar_categorias)

    for df in [df_train_proc, df_test_proc]:
        if 'tipo_do_aço_A300' in df.columns:
            df['tipo_do_aço_A300'] = df['tipo_do_aço_A300'].astype(str).str.strip().str.lower()
        if 'tipo_do_aço_A400' in df.columns and 'tipo_do_aço_A300' in df.columns:
            def tratar_na_tipo_aco(row):
                if pd.isna(row['tipo_do_aço_A400']):
                    if row['tipo_do_aço_A300'] == 'sim':
                        return 'Não'
                    elif row['tipo_do_aço_A300'] == 'não':
                        return 'Sim'
                return row['tipo_do_aço_A400']
            df['tipo_do_aço_A400'] = df.apply(tratar_na_tipo_aco, axis=1)

    return df_train_proc, df_test_proc

def normalizar_dados(df):
    colunas_numericas = df.select_dtypes(include=[float, int]).columns.tolist()
    scaler = MinMaxScaler()
    df_normalizado = df.copy()
    df_normalizado[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    return df_normalizado

def tratar_categoricas(df):
    colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    label_encoder = LabelEncoder()
    for coluna in colunas_categoricas:
        df[coluna] = label_encoder.fit_transform(df[coluna].astype(str))
    return df
