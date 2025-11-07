"""
Ponto de entrada principal para o projeto AV2 de IA.

Orquestra o carregamento dos dados, execução da validação cruzada
e apresentação dos resultados.
"""

import argparse
import numpy as np
import random
import sys
from pathlib import Path
from typing import Dict, Any

# --- Importações (sys.path) ---
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from carregar_data import carregar_dados_banknote
    from utils import configurar_sementes, executar_testes_utilitarios
    
    # (NOVAS IMPORTAÇÕES - ETAPA 6)
    from metricas import NOMES_METRICAS, NOMES_TEMPOS
    from cv import validacao_cruzada
    
    # (NOVAS IMPORTAÇÕES - Modelos)
    from modelo_knn import KNN
    from modelo_bayes_uni import BayesianoUnivariado
    from modelo_bayes_multi import BayesianoMultivariado
    
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Verifique se todos os arquivos .py estão na pasta 'src/'")
    sys.exit(1)


# --- Constantes ---
SEED = 42
DATA_FILE_PATH = Path(__file__).parent / "data" / "data_banknote_authentication.csv"


def parsear_argumentos() -> argparse.Namespace:
    """
    Configura e processa (parse) os argumentos da linha de comando (CLI).
    """
    parser = argparse.ArgumentParser(
        description="Executa validação cruzada 10-fold nos classificadores KNN e Bayes."
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=str(DATA_FILE_PATH),
        help=f"Caminho para o arquivo CSV do dataset. Padrão: {DATA_FILE_PATH}"
    )
    
    parser.add_argument(
        '--normalizar',
        action='store_true',
        help="Aplica normalização Z-score (StandardScaler) nos dados (fit no treino, transform no teste)."
    )
    
    parser.add_argument(
        '--knn_k_range',
        nargs='+',
        type=int,
        default=[1, 3, 5, 7, 9, 11],
        help="Lista de valores 'k' para testar na seleção de K do KNN. Padrão: 1 3 5 7 9 11"
    )
    
    parser.add_argument(
        '--bayes_multi_reg',
        type=float,
        default=1e-6,
        help="Valor 'epsilon' para regularização da matriz de covariância (Bayes Multi). Padrão: 1e-6"
    )
    
    return parser.parse_args()


def executar_avaliacao(args: argparse.Namespace):
    """
    Função principal que executa o fluxo de avaliação.
    """
    print(f"Iniciando avaliação com semente global = {SEED}")
    
    # 1. Configurar Sementes
    configurar_sementes(SEED)
    
    # 1.5. Executar testes rápidos
    executar_testes_utilitarios()

    # 2. Carregar Dados
    print(f"\nCarregando dados de: {args.data_path}")
    X, y = carregar_dados_banknote(Path(args.data_path))
    
    if X.size == 0 or y.size == 0:
        print("Erro: Não foi possível carregar os dados. Encerrando.")
        return 

    print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} atributos.")
    classes_unicas, contagem_classes = np.unique(y, return_counts=True)
    print(f"Distribuição das classes: {dict(zip(classes_unicas, contagem_classes))}")
    if args.normalizar:
        print("Info: Normalização Z-Score (StandardScaler) está ATIVADA.")
    else:
        print("Info: Normalização Z-Score (StandardScaler) está DESATIVADA.")

    # 3. Configurar Classificadores (AGORA IMPLEMENTADO)
    # Define a configuração dos 4 classificadores que vamos avaliar
    classificadores_config = {
        # KNNs são especiais: 'tipo: knn' aciona a seleção de K em cv.py
        "KNN (Euclidiana)": {
            'tipo': 'knn', 
            'distancia': 'euclidiana'
        },
        "KNN (Manhattan)": {
            'tipo': 'knn', 
            'distancia': 'manhattan'
        },
        # Bayes são padrão: 'classe' é a classe Python, 'params' são os kwargs
        "Bayesiano (Univariado)": {
            'tipo': 'padrao',
            'classe': BayesianoUnivariado,
            'params': {} # Sem parâmetros extras (usa o epsilon padrão)
        },
        "Bayesiano (Multivariado)": {
            'tipo': 'padrao',
            'classe': BayesianoMultivariado,
            'params': {'reg_cov': args.bayes_multi_reg} # Pega da CLI
        }
    }
    
    # 4. Executar Validação Cruzada (AGORA IMPLEMENTADO)
    # Esta função agora faz todo o trabalho pesado
    resultados_agregados = validacao_cruzada(X, y, classificadores_config, args)
    
    # 5. Apresentar Resultados (Tabela) (AGORA IMPLEMENTADO)
    print("\nAvaliação concluída. Gerando tabela de resultados...")
    imprimir_tabela_resultados(resultados_agregados)


def imprimir_tabela_resultados(resultados: Dict[str, Dict[str, Any]]):
    """
    Formata e imprime a tabela de resultados (Média ± Desvio Padrão).
    
    Args:
        resultados: Dicionário aninhado com resultados agregados por classificador.
    """
    
    # Define a ordem das colunas
    nomes_colunas = NOMES_METRICAS + NOMES_TEMPOS
    
    # Larguras das colunas (ajustadas)
    largura_nome = 30
    larguras_metricas = {
        "Acurácia": 15,
        "Precisão": 15,
        "F1-Score": 15,
        "Tempo Treino (s)": 18,
        "Tempo Teste (s)": 17
    }

    # --- Imprimir Cabeçalho ---
    header_list = [f"{'Classificador':<{largura_nome}}"]
    for nome_col in nomes_colunas:
        header_list.append(f"{nome_col:^{larguras_metricas[nome_col]}}")
    
    header = " | ".join(header_list)
    print("\n" + header)
    print("-" * len(header))

    # --- Imprimir Linhas (Resultados) ---
    # Garante a ordem solicitada no enunciado
    ordem_impressao = [
        "KNN (Euclidiana)", 
        "KNN (Manhattan)", 
        "Bayesiano (Multivariado)", # Coloquei Multi antes de Uni (geralmente é melhor)
        "Bayesiano (Univariado)"
    ]
    
    for nome_classificador in ordem_impressao:
        if nome_classificador not in resultados:
            continue # Ignora se o classificador não foi rodado
            
        metricas_clf = resultados[nome_classificador]
        linha_list = [f"{nome_classificador:<{largura_nome}}"]
        
        for nome_col in nomes_colunas:
            # Formato: "0.XXX ± 0.XXX"
            # (Usamos 4 casas decimais para métricas, 5 para tempo)
            if nome_col in NOMES_METRICAS:
                fmt = ".4f"
            else:
                fmt = ".5f" # Mais precisão para tempos
                
            media = metricas_clf[nome_col]['media']
            desvio = metricas_clf[nome_col]['desvio']
            
            texto_metrica = f"{media:{fmt}} ± {desvio:{fmt}}"
            linha_list.append(f"{texto_metrica:^{larguras_metricas[nome_col]}}")
            
        print(" | ".join(linha_list))
    
    print("-" * len(header))


if __name__ == "__main__":
    args = parsear_argumentos()
    executar_avaliacao(args)