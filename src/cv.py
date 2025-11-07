"""
Módulo de Validação Cruzada (Cross-Validation).

Implementa a lógica de divisão 10-fold estratificada, 
o loop de avaliação e a seleção de hiperparâmetros (K do KNN).
"""

import numpy as np
import argparse
from typing import Dict, List, Any, Generator, Tuple

# Importa nossas implementações
from utils import NormalizadorPadrao
from metricas import calcular_metricas, NOMES_METRICAS, NOMES_TEMPOS
from modelo_knn import KNN
from modelo_bayes_uni import BayesianoUnivariado
from modelo_bayes_multi import BayesianoMultivariado

# --- Constantes de CV ---
K_FOLD_PRINCIPAL = 10
# Usamos 3-fold para a seleção de K (um valor menor para ser rápido)
K_FOLD_INTERNO_KNN = 3 
SEED = 42 # Seed global do projeto


def _gerar_indices_folds_estratificados(y: np.ndarray, 
                                        k_fold: int, 
                                        seed: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Gerador manual do StratifiedKFold (do zero).
    
    Garante que a proporção das classes seja mantida em cada fold.
    
    Args:
        y (np.ndarray): Vetor de rótulos (n_samples,).
        k_fold (int): Número de folds (ex: 10).
        seed (int): Semente para embaralhamento reprodutível.

    Yields:
        (indices_treino, indices_teste): Tupla de arrays NumPy com os índices.
    """
    classes = np.unique(y)
    indices_por_classe: Dict[int, np.ndarray] = {}
    
    # 1. Separa os índices por classe
    for classe in classes:
        indices_por_classe[classe] = np.where(y == classe)[0]
        
    # 2. Embaralha os índices de cada classe (reprodutível)
    rng = np.random.default_rng(seed)
    for classe in classes:
        rng.shuffle(indices_por_classe[classe])
        
    # 3. Divide os índices embaralhados em 'k_fold' pedaços (chunks)
    folds_por_classe: Dict[int, List[np.ndarray]] = {}
    for classe in classes:
        # np.array_split lida com divisões não exatas
        folds_por_classe[classe] = np.array_split(indices_por_classe[classe], k_fold)

    # 4. Gera os folds (treino/teste)
    for i_fold in range(k_fold):
        indices_teste_list = []
        indices_treino_list = []
        
        # Para cada classe
        for classe in classes:
            # Pega o 'i-ésimo' chunk como teste
            indices_teste_list.append(folds_por_classe[classe][i_fold])
            
            # Pega todos os *outros* chunks como treino
            for j_fold in range(k_fold):
                if i_fold != j_fold:
                    indices_treino_list.append(folds_por_classe[classe][j_fold])
                    
        # Concatena os índices de todas as classes
        indices_teste = np.concatenate(indices_teste_list)
        indices_treino = np.concatenate(indices_treino_list)
        
        yield indices_treino, indices_teste


def _selecionar_melhor_k(X_treino_fold: np.ndarray, 
                         y_treino_fold: np.ndarray, 
                         k_range: List[int], 
                         distancia: str,
                         usar_normalizacao: bool) -> int:
    """
    Executa uma sub-validação cruzada (3-fold) DENTRO do conjunto 
    de treino principal para encontrar o melhor 'k' para o KNN.
    
    Métrica de seleção: F1-Score (Macro)
    """
    
    # print(f"  [KNN-Tune]: Iniciando seleção de K ({distancia})...")
    
    # Dicionário para armazenar o F1 médio de cada K
    f1_medio_por_k: Dict[int, float] = {}

    for k_val in k_range:
        f1_scores_do_k = [] # Armazena os F1s dos 3 folds internos
        
        # Gera os 3 folds internos (estratificados)
        # Usamos uma seed diferente (seed+1) para esta sub-divisão
        gen_folds_internos = _gerar_indices_folds_estratificados(y_treino_fold, K_FOLD_INTERNO_KNN, SEED + 1)
        
        for indices_treino_sub, indices_valid_sub in gen_folds_internos:
            
            X_treino_sub = X_treino_fold[indices_treino_sub]
            y_treino_sub = y_treino_fold[indices_treino_sub]
            X_valid_sub = X_treino_fold[indices_valid_sub]
            y_valid_sub = y_treino_fold[indices_valid_sub]
            
            # Normalização (se ativada), fit APENAS no sub-treino
            if usar_normalizacao:
                norm_sub = NormalizadorPadrao()
                X_treino_sub = norm_sub.fit_transform(X_treino_sub)
                X_valid_sub = norm_sub.transform(X_valid_sub)
            
            # Treina e avalia o K
            modelo_knn_sub = KNN(k=k_val, distancia=distancia)
            modelo_knn_sub.fit(X_treino_sub, y_treino_sub)
            y_pred_sub = modelo_knn_sub.predict(X_valid_sub)
            
            metricas_sub = calcular_metricas(y_valid_sub, y_pred_sub)
            f1_scores_do_k.append(metricas_sub["F1-Score"])
        
        # Calcula o F1 médio para este 'k_val'
        f1_medio_por_k[k_val] = np.mean(f1_scores_do_k)

    # Encontra o 'k' que teve o maior F1-Score médio
    melhor_k = max(f1_medio_por_k, key=f1_medio_por_k.get)
    # print(f"  [KNN-Tune]: Melhor K encontrado: {melhor_k} (F1: {f1_medio_por_k[melhor_k]:.4f})")
    
    return melhor_k


def _agregar_resultados(resultados_brutos: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
    """
    Calcula a média e o desvio padrão (std) dos resultados
    coletados nos K folds.
    """
    resultados_agregados = {}
    
    for nome_classificador, metricas_dict in resultados_brutos.items():
        resultados_agregados[nome_classificador] = {}
        for nome_metrica, lista_valores in metricas_dict.items():
            
            media = np.mean(lista_valores)
            desvio = np.std(lista_valores)
            
            resultados_agregados[nome_classificador][nome_metrica] = {
                "media": media,
                "desvio": desvio
            }
            
    return resultados_agregados


def validacao_cruzada(X: np.ndarray, 
                      y: np.ndarray, 
                      classificadores_config: Dict[str, Dict[str, Any]], 
                      args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """
    Função principal que orquestra o loop de validação cruzada 10-fold.
    
    Args:
        X: Todos os atributos.
        y: Todos os rótulos.
        classificadores_config: Dicionário vindo do main.py
        args: Argumentos da CLI (para normalização, k_range, etc.)

    Returns:
        Dicionário com os resultados agregados (media, desvio).
    """
    
    # Dicionário para armazenar as métricas de *cada* fold
    # Ex: {'KNN (Euclidiana)': {'Acurácia': [0.98, 0.95, ...], 'Tempo Treino (s)': [0.01, 0.02, ...]}}
    nomes_classificadores = list(classificadores_config.keys())
    todas_metricas = NOMES_METRICAS + NOMES_TEMPOS
    
    resultados_brutos: Dict[str, Dict[str, List[float]]] = {
        nome: {metrica: [] for metrica in todas_metricas}
        for nome in nomes_classificadores
    }
    
    # Armazena o K ótimo de cada fold (apenas para KNNs)
    k_otimos_por_fold: Dict[str, List[int]] = {
        nome: [] for nome, config in classificadores_config.items() if config['tipo'] == 'knn'
    }

    # --- Início do Loop 10-Fold ---
    print(f"\nIniciando Validação Cruzada ({K_FOLD_PRINCIPAL}-Folds Estratificados)...")
    
    gen_folds = _gerar_indices_folds_estratificados(y, K_FOLD_PRINCIPAL, SEED)
    
    for i_fold, (indices_treino, indices_teste) in enumerate(gen_folds):
        
        print(f"  Processando Fold {i_fold + 1}/{K_FOLD_PRINCIPAL}...")
        
        # 1. Separar dados do fold
        X_treino, y_treino = X[indices_treino], y[indices_treino]
        X_teste, y_teste = X[indices_teste], y[indices_teste]
        
        # 2. Normalização (se ativada)
        if args.normalizar:
            normalizador = NormalizadorPadrao()
            # Fit APENAS no treino
            X_treino = normalizador.fit_transform(X_treino)
            # Transform no treino e no teste
            X_teste = normalizador.transform(X_teste)
            
        # 3. Loop de Classificadores
        for nome, config in classificadores_config.items():
            
            # --- Bloco de Treinamento ---
            if config['tipo'] == 'knn':
                # 3.A.1 (KNN): Selecionar K ótimo (usando 3-fold interno)
                k_otimo = _selecionar_melhor_k(
                    X_treino, y_treino, 
                    args.knn_k_range, 
                    config['distancia'], 
                    args.normalizar
                )
                k_otimos_por_fold[nome].append(k_otimo)
                
                # 3.A.2 (KNN): Treinar modelo final do fold com K ótimo
                modelo = KNN(k=k_otimo, distancia=config['distancia'])
                
            else:
                # 3.B (Bayes): Instanciar modelo
                modelo = config['classe'](**config['params'])
            
            # 3.C (Todos): Treinar
            # (o decorador @medir_tempo salva o modelo.tempo_treino_)
            modelo.fit(X_treino, y_treino)

            # --- Bloco de Teste ---
            # (o decorador @medir_tempo salva o modelo.tempo_teste_)
            y_pred = modelo.predict(X_teste)
            
            # 4. Calcular e armazenar métricas
            metricas_fold = calcular_metricas(y_teste, y_pred)
            
            for metrica_nome in NOMES_METRICAS:
                resultados_brutos[nome][metrica_nome].append(metricas_fold[metrica_nome])
                
            resultados_brutos[nome][NOMES_TEMPOS[0]].append(modelo.tempo_treino_)
            resultados_brutos[nome][NOMES_TEMPOS[1]].append(modelo.tempo_teste_)
            
    print("... Validação Cruzada concluída.")
    
    # 5. Imprimir média dos Ks ótimos (informativo)
    for nome_knn, k_lista in k_otimos_por_fold.items():
        print(f"  [Info] K médio escolhido para {nome_knn}: {np.mean(k_lista):.2f} (de {args.knn_k_range})")

    # 6. Agregar resultados (calcular média e desvio)
    return _agregar_resultados(resultados_brutos)