📘 Projeto: Classificação Banknote Authentication (IA)

Este projeto implementa, do zero, três classificadores supervisionados utilizando apenas Python e NumPy — sem scikit-learn ou pandas:

✅ K-Nearest Neighbors (KNN)
✅ Naive Bayes Gaussiano Univariado
✅ Naive Bayes Gaussiano Multivariado

O objetivo é analisar o desempenho desses modelos no dataset Banknote Authentication da UCI, incluindo validação cruzada estratificada e comparação de resultados.

📂 Estrutura do Projeto
/
|-- main.py                       # Ponto de entrada da aplicação (CLI + execução completa)
|-- README.md                     # Este arquivo
|-- .gitignore
|-- data/
|   └── data_banknote_authentication.csv   # (Baixar manualmente)
|-- src/
    |-- carregar_data.py          # Leitura e pré-processamento do CSV
    |-- metricas.py               # Métricas: Acurácia, Precisão, F1-score
    |-- modelo_knn.py             # Implementação do algoritmo KNN
    |-- modelo_bayes_uni.py       # Naive Bayes Gaussiano Univariado
    |-- modelo_bayes_multi.py     # Naive Bayes Gaussiano Multivariado
    |-- cv.py                     # Validação Cruzada Estratificada (10-fold)
    |-- timing.py                 # Decorador @medir_tempo
    |-- utils.py                  # Normalização, semente e utilitários

🧪 Pré-Requisitos

Python 3.9+

Pip

Git (opcional)

⚙️ Setup do Ambiente
1️⃣ Clone o repositório (opcional)
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO

2️⃣ Crie um ambiente virtual
python -m venv .venv

3️⃣ Ative o ambiente

Linux/macOS:

source .venv/bin/activate


Windows (CMD):

.\.venv\Scripts\activate


Windows (PowerShell):

.\.venv\Scripts\Activate.ps1

4️⃣ Instale as dependências
pip install numpy

📥 Baixar o Dataset

Acesse: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Clique em "Data Folder"

Baixe o arquivo: data_banknote_authentication.csv

Coloque o arquivo em: data/ na raiz do projeto

🚀 Execução

O arquivo main.py:

Carrega os dados

Executa validação cruzada estratificada (10-fold)

Seleciona o melhor K para o KNN

Compara todos os classificadores

Exibe a tabela final de resultados

▶️ Rodar com configuração padrão
python main.py

🧮 Rodar com normalização Z-Score
python main.py --normalizar

🛠️ Argumentos da CLI
Argumento	Descrição	Exemplo
--knn_k_range [K ...]	Lista de valores de K para testar no KNN	python main.py --knn_k_range 1 3 5
--bayes_multi_reg [float]	Valor de regularização (ε) do Bayes Multivariado (padrão: 1e-6)	python main.py --bayes_multi_reg 1e-5
