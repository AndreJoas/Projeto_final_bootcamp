# Aplicação de Classificação com Otimização de Modelos
Esta aplicação web utiliza Flask para fornecer uma interface interativa para o treinamento, 
otimização e avaliação de diferentes modelos de machine learning. O objetivo é realizar previsões de falhas industriais a partir de dados 
de treinamento e teste, e fornecer estatísticas e análises dos resultados.

1# instalação de dependecias
  pip install -r requirements.txt
2#estrutura do projeto
  
├── app.py                   # Arquivo principal da aplicação Flask
├── requirements.txt         # Dependências do projeto
├── templates/               # Templates HTML
│   ├── index.html           # Página inicial com gráficos e tabelas
│   ├── otimizar_modelo.html # Formulário para otimização de modelos
│   └── detalhes.html        # Página de detalhes das previsões
├── static/                  # Arquivos estáticos como CSS e JS
│   └── css/
├── modelo/                  # Modelos de machine learning
├── dados/                   # Dados de entrada (treinamento e teste)
├── estatistica.py           # Funções para gerar dados estatísticos
└── processamento/           # Scripts para pipeline de pré-processamento de dados

#Rodando a Aplicação
1. Preparar os Dados
   Antes de rodar a aplicação verifique se tem bases de treino e teste na pasta "base" e com os nomes CORRETOS
   Antes de rodar a aplicação, você precisa garantir que os dados estejam no formato correto. A aplicação espera os seguintes arquivos CSV:

basetratada.csv: Dados de treino tratados.

basetest.csv: Dados de teste tratados.

Você pode gerar esses arquivos utilizando o script pipeline_completo() localizado no módulo processamento/pipeline.py. O script irá ler e tratar os dados, salvando os arquivos CSV necessários.
So basta rodar a aplicação que ela mesma gerar essas bases tratadas

2. Treinamento dos Modelos
A aplicação inclui a função treinar_e_salvar_modelos(), que treina diversos modelos de
aprendizado de máquina (como HistGradientBoosting, LGBMClassifier, XGBClassifier, etc.) utilizando os dados de treino e salva as previsões em arquivos CSV. (So basta rodar a aplicação )

3. Acessando a Interface Web
A interface possui as seguintes rotas principais:

/: Página inicial com gráficos e estatísticas geradas a partir dos dados de entrada.

/detalhes: Página de detalhes das previsões geradas pelos modelos, com a possibilidade de baixar os arquivos CSV gerados.

/treinar_modelo_form: Formulário para otimizar os parâmetros dos modelos utilizando Optuna.

4. Baixar Previsões
Após treinar os modelos, as previsões podem ser baixadas em formato CSV na página /detalhes. O arquivo CSV contém as previsões de falha para cada modelo treinado.
