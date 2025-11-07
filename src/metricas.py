"""
Módulo para cálculo das métricas de avaliação (do zero).

Inclui:
- Matriz de Confusão
- Acurácia
- Precisão (Macro)
- Recall (Macro) - Necessário para o F1
- F1-Score (Macro)
"""

import numpy as np
from typing import Dict, List, Tuple

# Constantes que serão importadas pelo cv.py e main.py
NOMES_METRICAS = ["Acurácia", "Precisão", "F1-Score"]
NOMES_TEMPOS = ["Tempo Treino (s)", "Tempo Teste (s)"]
# Epsilon para evitar divisão por zero em precisão/recall
EPSILON = 1e-9


def calcular_matriz_confusao(y_real: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de confusão.
    
    Args:
        y_real: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        classes: Array de classes únicas (ex: [0, 1]).

    Returns:
        Matriz de confusão (np.ndarray) de shape (n_classes, n_classes).
        Onde [i, j] é o número de amostras da classe 'i' (real) preditas como 'j' (predição).
    """
    num_classes = len(classes)
    
    # Cria um mapa de {valor_classe: indice_matriz}
    # Ex: se classes=[0, 1], mapa={0: 0, 1: 1}
    mapa_classes = {classe: i for i, classe in enumerate(classes)}
    
    # Inicializa a matriz com zeros
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    
    # Preenche a matriz
    for i in range(len(y_real)):
        indice_real = mapa_classes[y_real[i]]
        indice_pred = mapa_classes[y_pred[i]]
        matriz[indice_real, indice_pred] += 1
        
    return matriz


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula o conjunto de métricas (Acurácia, Precisão Macro, F1 Macro).
    
    Args:
        y_real: Rótulos verdadeiros.
        y_pred: Rótulos preditos.

    Returns:
        Dicionário com os nomes das métricas e seus valores.
    """
    
    # Garante que as classes sejam [0, 1] ou as classes presentes nos dados reais
    classes = np.unique(y_real)
    num_classes = len(classes)
    
    if num_classes == 0 or len(y_real) == 0:
        return {"Acurácia": 0.0, "Precisão": 0.0, "F1-Score": 0.0}

    # 1. Calcular Matriz de Confusão
    matriz = calcular_matriz_confusao(y_real, y_pred, classes)
    
    # 2. Calcular Acurácia
    # Acurácia = (Soma da diagonal principal) / (Total de amostras)
    corretos = np.diag(matriz).sum()
    total = matriz.sum()
    acuracia = corretos / total if total > 0 else 0.0
    
    # 3. Calcular Precisão e F1 (Macro-Averaging)
    # Precisamos calcular por classe e depois tirar a média
    
    precisoes_por_classe = []
    recalls_por_classe = []
    f1s_por_classe = []
    
    for i in range(num_classes):
        # Verdadeiros Positivos (VP) para a classe 'i'
        VP = matriz[i, i]
        
        # Falsos Positivos (FP) para a classe 'i'
        # É a soma da *coluna* 'i' (tudo que foi predito como 'i'), menos o VP
        FP = matriz[:, i].sum() - VP
        
        # Falsos Negativos (FN) para a classe 'i'
        # É a soma da *linha* 'i' (tudo que era realmente 'i'), menos o VP
        FN = matriz[i, :].sum() - VP
        
        # Calcular Precisão e Recall da classe 'i'
        # Usamos EPSILON para evitar divisão por zero
        precisao = VP / (VP + FP + EPSILON)
        recall = VP / (VP + FN + EPSILON)
        
        # Calcular F1-Score da classe 'i'
        f1 = 2 * (precisao * recall) / (precisao + recall + EPSILON)
        
        precisoes_por_classe.append(precisao)
        recalls_por_classe.append(recall) # Não pedido, mas necessário para F1
        f1s_por_classe.append(f1)
        
    # Calcular a média (Macro)
    # np.mean lida com a média de listas
    precisao_macro = np.mean(precisoes_por_classe)
    f1_macro = np.mean(f1s_por_classe)
    
    return {
        "Acurácia": acuracia,
        "Precisão": precisao_macro, # Conforme solicitado: Precisão (macro)
        "F1-Score": f1_macro      # Conforme solicitado: F1-Score (macro)
    }