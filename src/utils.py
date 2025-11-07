"""
Módulo de utilitários diversos:
... (resto da docstring igual) ...
"""

import numpy as np
import random
from typing import Tuple

# Importa as funções que vamos testar
from metricas import calcular_metricas
from modelo_knn import KNN
from modelo_bayes_uni import BayesianoUnivariado
from modelo_bayes_multi import BayesianoMultivariado


def configurar_sementes(seed: int = 42):
    # ... (função igual) ...
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Utils]: Sementes (random, numpy) configuradas para {seed}")

class NormalizadorPadrao:
    # ... (classe igual) ...
    def __init__(self, epsilon: float = 1e-9):
        self.media_: np.ndarray | None = None
        self.desvio_padrao_: np.ndarray | None = None
        self.epsilon = epsilon
    def fit(self, X: np.ndarray):
        self.media_ = np.mean(X, axis=0)
        self.desvio_padrao_ = np.std(X, axis=0)
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.media_ is None or self.desvio_padrao_ is None:
            raise RuntimeError("O Normalizador deve ser 'fit' antes de 'transform'.")
        return (X - self.media_) / (self.desvio_padrao_ + self.epsilon)
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def _testar_knn():
    # ... (função igual) ...
    print("[Utils-Teste]: Testando Modelo KNN...")
    X_treino_0 = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_treino_0 = np.array([0, 0, 0, 0])
    X_treino_1 = np.array([[5,5], [5,6], [6,5], [6,6]])
    y_treino_1 = np.array([1, 1, 1, 1])
    X_treino = np.vstack([X_treino_0, X_treino_1])
    y_treino = np.hstack([y_treino_0, y_treino_1])
    X_teste = np.array([[0.5, 0.5], [5.5, 5.5]])
    y_real = np.array([0, 1])
    
    knn_k1_euclid = KNN(k=1, distancia='euclidiana')
    knn_k1_euclid.fit(X_treino, y_treino)
    y_pred_k1_e = knn_k1_euclid.predict(X_teste)
    assert np.array_equal(y_real, y_pred_k1_e), "Falha KNN (K=1, Euclidiana)"
    
    knn_k3_manhat = KNN(k=3, distancia='manhattan')
    knn_k3_manhat.fit(X_treino, y_treino)
    y_pred_k3_m = knn_k3_manhat.predict(X_teste)
    assert np.array_equal(y_real, y_pred_k3_m), "Falha KNN (K=3, Manhattan)"
    
    X_teste_empate = np.array([[2.5, 2.5]])
    knn_k2_euclid = KNN(k=2, distancia='euclidiana')
    knn_k2_euclid.fit(X_treino, y_treino) # y_treino é [0,0,0,0,1,1,1,1]
    y_pred_k2_e = knn_k2_euclid.predict(X_teste_empate)
    assert y_pred_k2_e[0] == 0, "Falha KNN (K=2, Desempate)"
    print("[Utils-Teste]: ... Modelo KNN OK.")

# -------------------------------------------------------------------
# (NOVA FUNÇÃO DE TESTE)
# Adicione esta função *antes* de 'executar_testes_utilitarios'
# -------------------------------------------------------------------
def _testar_bayes_uni():
    """Teste rápido e sintético para o modelo Bayesiano Univariado."""
    print("[Utils-Teste]: Testando Bayesiano Univariado...")
    
    # 1. Usar os mesmos dados sintéticos do KNN
    X_treino_0 = np.array([[0,0], [0,1], [1,0], [1,1]]) # Média [0.5, 0.5]
    y_treino_0 = np.array([0, 0, 0, 0])
    X_treino_1 = np.array([[5,5], [5,6], [6,5], [6,6]]) # Média [5.5, 5.5]
    y_treino_1 = np.array([1, 1, 1, 1])
    
    X_treino = np.vstack([X_treino_0, X_treino_1])
    y_treino = np.hstack([y_treino_0, y_treino_1])
    
    # Pontos de teste: exatamente nas médias
    X_teste = np.array([[0.5, 0.5], [5.5, 5.5], [0.6, 0.6], [5.4, 5.4]])
    y_real = np.array([0, 1, 0, 1]) # O modelo deve acertar
    
    # 2. Testar o modelo
    bayes_uni = BayesianoUnivariado()
    bayes_uni.fit(X_treino, y_treino)
    
    # 3. Checar parâmetros (sanity check)
    assert np.allclose(bayes_uni.log_priors_, np.log(0.5)), "Falha BayesUni (Priors)"
    assert np.allclose(bayes_uni.media_[0, :], [0.5, 0.5]), "Falha BayesUni (Média C0)"
    assert np.allclose(bayes_uni.media_[1, :], [5.5, 5.5]), "Falha BayesUni (Média C1)"
    # Variância de [0,0,1,1] é 0.25
    assert np.allclose(bayes_uni.variancia_[0, :], 0.25 + bayes_uni.var_epsilon), "Falha BayesUni (Var C0)"
    assert np.allclose(bayes_uni.variancia_[1, :], 0.25 + bayes_uni.var_epsilon), "Falha BayesUni (Var C1)"

    # 4. Testar predição
    y_pred = bayes_uni.predict(X_teste)
    assert np.array_equal(y_real, y_pred), "Falha BayesUni (Predição)"
    
    print("[Utils-Teste]: ... Bayesiano Univariado OK.")
# -------------------------------------------------------------------

def _testar_bayes_multi():
    """Teste rápido e sintético para o modelo Bayesiano Multivariado."""
    print("[Utils-Teste]: Testando Bayesiano Multivariado...")
    
    # 1. Usar os mesmos dados sintéticos
    X_treino_0 = np.array([[0,0], [0,1], [1,0], [1,1]]) # Média [0.5, 0.5]
    y_treino_0 = np.array([0, 0, 0, 0])
    X_treino_1 = np.array([[5,5], [5,6], [6,5], [6,6]]) # Média [5.5, 5.5]
    y_treino_1 = np.array([1, 1, 1, 1])
    
    X_treino = np.vstack([X_treino_0, X_treino_1])
    y_treino = np.hstack([y_treino_0, y_treino_1])
    
    # Pontos de teste: exatamente nas médias
    X_teste = np.array([[0.5, 0.5], [5.5, 5.5], [0.6, 0.6], [5.4, 5.4]])
    y_real = np.array([0, 1, 0, 1])
    
    # 2. Testar o modelo
    # Usamos uma regularização maior (reg_cov=0.1) só para o teste
    # pois os dados sintéticos têm poucas amostras
    bayes_multi = BayesianoMultivariado(reg_cov=0.1) 
    bayes_multi.fit(X_treino, y_treino)
    
    # 3. Checar parâmetros (sanity check)
    assert np.isclose(bayes_multi.log_priors_[0], np.log(0.5)), "Falha BayesMulti (Priors C0)"
    assert np.isclose(bayes_multi.log_priors_[1], np.log(0.5)), "Falha BayesMulti (Priors C1)"
    assert np.allclose(bayes_multi.media_[0], [0.5, 0.5]), "Falha BayesMulti (Média C0)"
    assert np.allclose(bayes_multi.media_[1], [5.5, 5.5]), "Falha BayesMulti (Média C1)"

    # Covariância de [0,0,1,1] e [0,1,0,1] é [[0.25, 0], [0, 0.25]]
    cov_esperada_0 = np.array([[0.25, 0], [0, 0.25]])
    # Acessamos a inversa da (cov + reg*I)
    inv_cov_esperada_0 = np.linalg.inv(cov_esperada_0 + np.eye(2) * 0.1)
    
    assert np.allclose(bayes_multi.inv_cov_[0], inv_cov_esperada_0), "Falha BayesMulti (InvCov C0)"

    # 4. Testar predição
    y_pred = bayes_multi.predict(X_teste)
    assert np.array_equal(y_real, y_pred), "Falha BayesMulti (Predição)"
    
    print("[Utils-Teste]: ... Bayesiano Multivariado OK.")
# -----------------------------------------------------------------


def executar_testes_utilitarios():
    """
    Executa testes rápidos (sanity checks)...
    """
    print("\n[Utils]: Executando testes rápidos (sanity checks)...")
    
    try:
        # ... (Testes 1, 2, 3 permanecem iguais) ...
        
        print("[Utils-Teste]: Testando NormalizadorPadrao...")
        X_teste_norm = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        normalizador = NormalizadorPadrao()
        normalizador.fit(X_teste_norm)
        media_esperada = np.array([2.0, 20.0])
        std_esperado = np.array([np.std([1.0, 2.0, 3.0]), np.std([10.0, 20.0, 30.0])])
        assert np.allclose(normalizador.media_, media_esperada), "Falha no cálculo da Média"
        assert np.allclose(normalizador.desvio_padrao_, std_esperado), "Falha no cálculo do Desvio Padrão"
        X_transformado = normalizador.transform(X_teste_norm)
        assert np.allclose(np.mean(X_transformado, axis=0), 0.0), "Falha na Média pós-transformação"
        assert np.allclose(np.std(X_transformado, axis=0), 1.0), "Falha no STD pós-transformação"
        print("[Utils-Teste]: ... NormalizadorPadrao OK.")

        print("[Utils-Teste]: Testando Métricas (Caso Perfeito)...")
        y_real_perf = np.array([0, 0, 1, 1, 2, 2])
        y_pred_perf = np.array([0, 0, 1, 1, 2, 2])
        metricas_perf = calcular_metricas(y_real_perf, y_pred_perf)
        assert np.isclose(metricas_perf["Acurácia"], 1.0), "Falha Acurácia (Perfeito)"
        assert np.isclose(metricas_perf["Precisão"], 1.0), "Falha Precisão (Perfeito)"
        assert np.isclose(metricas_perf["F1-Score"], 1.0), "Falha F1-Score (Perfeito)"
        print("[Utils-Teste]: ... Métricas (Caso Perfeito) OK.")

        print("[Utils-Teste]: Testando Métricas (Caso Binário Imperfeito)...")
        y_real_bin = np.array([0, 1, 0, 1, 0, 1])
        y_pred_bin = np.array([0, 1, 1, 0, 0, 1])
        metricas_bin = calcular_metricas(y_real_bin, y_pred_bin)
        assert np.isclose(metricas_bin["Acurácia"], 4/6), "Falha Acurácia (Imperfeito)"
        assert np.isclose(metricas_bin["Precisão"], 2/3), "Falha Precisão (Imperfeito)"
        assert np.isclose(metricas_bin["F1-Score"], 2/3), "Falha F1-Score (Imperfeito)"
        print("[Utils-Teste]: ... Métricas (Caso Binário Imperfeito) OK.")
        
        _testar_knn()
        
        _testar_bayes_uni()

        _testar_bayes_multi()
        

        print("\n[Utils]: Todos os testes rápidos passaram com sucesso!")

    except AssertionError as e:
        print(f"\n[Utils]: FALHA NO TESTE RÁPIDO: {e}")
        # Encerra o programa se os utilitários básicos falharem
        exit(1)