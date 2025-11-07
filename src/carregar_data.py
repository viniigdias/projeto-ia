"""
Módulo responsável pelo carregamento e pré-processamento básico do dataset.
"""

import csv
import numpy as np
from typing import Tuple
from pathlib import Path

def carregar_dados_banknote(caminho_arquivo: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega o dataset Banknote Authentication de um arquivo CSV.
    
    Assume que o CSV não tem cabeçalho e a última coluna é a classe.
    
    Args:
        caminho_arquivo: Caminho (Path) para o arquivo .csv (ou .data).

    Returns:
        Uma tupla (X, y):
        X (np.ndarray): Matriz de atributos (n_samples, n_features).
        y (np.ndarray): Vetor de classes (n_samples,).
    """
    
    print(f"[carregar_data]: Tentando carregar dados de {caminho_arquivo}...")
    
    dados = []
    try:
        # Abre o arquivo CSV
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            leitor_csv = csv.reader(f)
            # Lê linha por linha
            for linha in leitor_csv:
                if linha: # Ignora linhas em branco
                    # Converte todos os valores da linha para float
                    dados.append([float(valor) for valor in linha])
                    
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo não encontrado em {caminho_arquivo}")
        print("Por favor, baixe o dataset e coloque-o na pasta 'data/' conforme o README.md")
        # Retorna arrays vazios se o arquivo não existir
        return np.array([]), np.array([])
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        # Retorna arrays vazios em caso de outro erro de leitura
        return np.array([]), np.array([])

    if not dados:
        print("Erro: O arquivo de dados está vazio.")
        return np.array([]), np.array([])

    # Converte a lista de listas para um array NumPy
    dados_np = np.array(dados)
    
    # Separa os atributos (X) das classes (y)
    # X são todas as colunas, exceto a última
    X = dados_np[:, :-1]
    # y é apenas a última coluna, convertida para inteiro (classes 0 e 1)
    y = dados_np[:, -1].astype(int) 
    
    print(f"[carregar_data]: Dados carregados com sucesso. Shape X: {X.shape}, Shape y: {y.shape}")
    
    return X, y