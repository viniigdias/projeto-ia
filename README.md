
Projeto - Classificação Banknote Authentication (IA)
Este projeto implementa e avalia classificadores KNN e Naive Bayes (Univariado e Multivariado) do zero, utilizando apenas Python e NumPy, para o dataset Banknote Authentication da UCI.

O código foi desenvolvido sem o uso de bibliotecas como scikit-learn ou pandas, focando na implementação manual dos algoritmos e da lógica de validação.

Estrutura do Projeto
/
|-- main.py             # Ponto de entrada principal, CLI, orquestração
|-- README.md           # Este arquivo
|-- .gitignore          # Ignora arquivos desnecessários (ex: .venv)
|-- data/
|   |-- data_banknote_authentication.csv # (Deve ser baixado manualmente)
|-- src/
    |-- carregar_data.py  # Carregamento e pré-processamento do CSV
    |-- metricas.py       # Funções de métricas (Acurácia, Precisão, F1)
    |-- modelo_knn.py     # Implementação do K-Nearest Neighbors
    |-- modelo_bayes_uni.py# Implementação do Bayes Gaussiano Univariado
    |-- modelo_bayes_multi.py# Implementação do Bayes Gaussiano Multivariado
    |-- cv.py             # Lógica de Validação Cruzada (10-fold estratificado)
    |-- timing.py         # Decorador @medir_tempo
    |-- utils.py          # Normalizador, sementes e testes de sanidade
Setup do Ambiente
Clone o repositório (opcional, se já baixou):

Bash

git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO
Crie um ambiente virtual:

Bash

python -m venv .venv
Ative o ambiente:

Linux/macOS:

Bash

source .venv/bin/activate
Windows (CMD):

Bash

.\.venv\Scripts\activate
Windows (PowerShell):

Bash

.\.venv\Scripts\Activate.ps1
Instale as dependências:

Bash

pip install numpy
Baixe o Dataset:

Acesse: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Clique em "Data Folder".

Baixe o arquivo data_banknote_authentication.csv.

Salve o arquivo na pasta data/ (na raiz do projeto).

Execução
O script principal main.py roda todos os testes, carrega os dados, executa a validação cruzada 10-fold completa (incluindo a seleção de K para o KNN) e imprime a tabela de resultados.

Comando Padrão (Sem normalização):

Bash

python main.py
Comando Opcional (Com normalização Z-Score):

Bash

python main.py --normalizar
Outros Argumentos da CLI:
--knn_k_range [K ...]

Define a lista de K's para a seleção (Padrão: 1 3 5 7 9 11).

Ex: python main.py --knn_k_range 1 3 5

--bayes_multi_reg [float]

Define o epsilon de regularização do Bayes Multi (Padrão: 1e-6).