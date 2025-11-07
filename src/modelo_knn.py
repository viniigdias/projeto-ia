"""
Módulo que implementa o classificador K-Nearest Neighbors (KNN) do zero.
"""

import numpy as np
from collections import Counter
from typing import Literal

# Importa nosso decorador de tempo
from timing import medir_tempo

class KNN:
    """
    Implementação do classificador K-Nearest Neighbors (KNN) do zero.
    
    Parâmetros:
    -----------
    k (int): 
        O número de vizinhos a considerar.
    distancia (str, 'euclidiana' ou 'manhattan'): 
        A métrica de distância a ser usada.
        
    Atributos (pós-fit):
    --------------------
    X_treino (np.ndarray): Dados de treino armazenados.
    y_treino (np.ndarray): Rótulos de treino armazenados.
    classe_majoritaria_treino_ (any): 
        Classe mais frequente no treino, usada para desempate.
    tempo_treino_ (float): Tempo (s) do fit, injetado pelo @medir_tempo.
    tempo_teste_ (float): Tempo (s) do predict, injetado pelo @medir_tempo.
    """
    
    def __init__(self, k: int = 3, distancia: Literal['euclidiana', 'manhattan'] = 'euclidiana'):
        if k < 1:
            raise ValueError("K (número de vizinhos) deve ser no mínimo 1.")
        
        self.k = k
        self.distancia = distancia
        
        # Atributos que serão definidos no 'fit'
        self.X_treino: np.ndarray | None = None
        self.y_treino: np.ndarray | None = None
        self.classe_majoritaria_treino_ = None
        
        # Atributos que serão definidos pelo decorador @medir_tempo
        self.tempo_treino_ = 0.0
        self.tempo_teste_ = 0.0

    @medir_tempo
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o classificador KNN.
        
        Para o KNN, "treinar" é simplesmente armazenar os dados 
        de treino na memória.
        """
        self.X_treino = X
        self.y_treino = y
        
        # Pré-calcula a classe mais comum do treino.
        # Isso é usado para quebrar empates durante a votação.
        if y.size > 0:
            self.classe_majoritaria_treino_ = Counter(y).most_common(1)[0][0]
        else:
            self.classe_majoritaria_treino_ = 0 # Um padrão, caso y esteja vazio

    def _calcular_distancias(self, X_teste: np.ndarray) -> np.ndarray:
        """
        Calcula a matriz de distâncias (vetorizada) entre
        cada ponto de teste e cada ponto de treino.
        
        Args:
            X_teste (np.ndarray): (n_amostras_teste, n_features)

        Returns:
            np.ndarray: Matriz (n_amostras_teste, n_amostras_treino)
        """
        if self.X_treino is None:
            raise RuntimeError("O modelo KNN deve ser treinado (fit) antes de prever (predict).")
            
        # --- Distância Euclidiana (L2) ---
        # (a - b)^2 = a^2 - 2ab + b^2
        if self.distancia == 'euclidiana':
            # X_teste shape: (n_t, f)
            # X_treino shape: (n_tr, f)
            
            # (n_t, 1)
            teste_sq = np.sum(X_teste**2, axis=1, keepdims=True)
            # (n_tr,) -> broadcast para (1, n_tr)
            treino_sq = np.sum(self.X_treino**2, axis=1)
            # (n_t, n_tr)
            dot_prod = X_teste @ self.X_treino.T
            
            # Broadcasting: (n_t, 1) + (1, n_tr) - (n_t, n_tr) -> (n_t, n_tr)
            dist_sq = teste_sq - 2 * dot_prod + treino_sq
            
            # Corrige pequenos erros numéricos (ex: -1e-15) antes do sqrt
            dist_sq = np.maximum(0, dist_sq)
            return np.sqrt(dist_sq)

        # --- Distância de Manhattan (L1) ---
        # sum(|a - b|)
        elif self.distancia == 'manhattan':
            # Usamos broadcasting avançado:
            # (n_t, f) -> (n_t, 1, f)
            # (n_tr, f) -> (1, n_tr, f)
            # Subtração (n_t, 1, f) - (1, n_tr, f) -> (n_t, n_tr, f)
            diff = X_teste[:, np.newaxis, :] - self.X_treino[np.newaxis, :, :]
            
            # Soma ao longo do último eixo (as features)
            return np.sum(np.abs(diff), axis=2) # Shape final: (n_t, n_tr)
        
        else:
            raise ValueError(f"Distância '{self.distancia}' não reconhecida. Use 'euclidiana' ou 'manhattan'.")

    def _votar(self, k_vizinhos_labels: np.ndarray) -> np.ndarray:
        """
        Realiza a votação para um conjunto de amostras de teste.
        
        Args:
            k_vizinhos_labels (np.ndarray): 
                Array 2D (n_amostras_teste, k) contendo os rótulos
                dos k vizinhos mais próximos para cada amostra de teste.

        Returns:
            np.ndarray: Array 1D (n_amostras_teste,) com a predição final.
        """
        predicoes_finais = []
        
        # Itera sobre cada *linha* (cada amostra de teste)
        for labels_amostra in k_vizinhos_labels:
            
            # 1. Conta os votos
            # Ex: labels_amostra = [0, 1, 1] -> Counter({1: 2, 0: 1})
            contagem_votos = Counter(labels_amostra)
            
            # 2. Obtém a lista ordenada de votos
            # Ex: [(1, 2), (0, 1)]  (classe 1 teve 2 votos, classe 0 teve 1)
            mais_comuns = contagem_votos.most_common()
            
            # 3. Verifica se há empate
            # Se (temos mais de 1 classe) E (votos do 1º == votos do 2º)
            if len(mais_comuns) > 1 and mais_comuns[0][1] == mais_comuns[1][1]:
                
                # EMPATE! Usa a classe majoritária do *treino* como desempate
                predicoes_finais.append(self.classe_majoritaria_treino_)
            else:
                # Sem empate, pega a classe que venceu (índice [0][0])
                predicoes_finais.append(mais_comuns[0][0])
                
        return np.array(predicoes_finais)

    @medir_tempo
    def predict(self, X_teste: np.ndarray) -> np.ndarray:
        """
        Prevê os rótulos para um novo conjunto de dados X_teste.
        """
        
        # 1. Calcular a matriz (n_teste, n_treino) de distâncias
        distancias = self._calcular_distancias(X_teste)
        
        # 2. Obter os *índices* dos K vizinhos mais próximos
        # np.argsort ordena e retorna os *índices*
        # Pegamos os primeiros 'k' índices por linha (axis=1)
        # Shape: (n_teste, k)
        indices_k_vizinhos = np.argsort(distancias, axis=1)[:, :self.k]
        
        # 3. Obter os *rótulos* desses vizinhos
        # Usamos indexação avançada do NumPy
        # Shape: (n_teste, k)
        k_vizinhos_labels = self.y_treino[indices_k_vizinhos]
        
        # 4. Realizar a votação
        # Shape final: (n_teste,)
        return self._votar(k_vizinhos_labels)