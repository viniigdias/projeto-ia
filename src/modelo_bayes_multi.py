"""
Módulo que implementa o classificador Bayesiano Gaussiano Multivariado.

Assume que cada classe segue uma distribuição Gaussiana Multivariada,
capturando as correlações entre os atributos.
"""

import numpy as np
from timing import medir_tempo
from typing import Dict

class BayesianoMultivariado:
    """
    Implementação do classificador Bayesiano Gaussiano Multivariado do zero.
    
    Parâmetros:
    -----------
    reg_cov (float): 
        Valor 'epsilon' (regularização) adicionado à diagonal da
        matriz de covariância para garantir estabilidade numérica
        e invertibilidade (Σ + εI).
        
    Atributos (pós-fit):
    --------------------
    classes_ (np.ndarray): 
        Array com os rótulos das classes (ex: [0, 1]).
    log_priors_ (Dict[int, float]): 
        Dicionário {classe: log_prior}
    media_ (Dict[int, np.ndarray]): 
        Dicionário {classe: vetor_media (n_features,)}
    
    # Parâmetros pré-calculados para eficiência no 'predict'
    inv_cov_ (Dict[int, np.ndarray]): 
        Dicionário {classe: matriz_covariancia_inversa (n_features, n_features)}
    log_det_cov_ (Dict[int, float]): 
        Dicionário {classe: log_determinante_matriz_covariancia}

    tempo_treino_ (float): Tempo (s) do fit, injetado pelo @medir_tempo.
    tempo_teste_ (float): Tempo (s) do predict, injetado pelo @medir_tempo.
    """
    
    def __init__(self, reg_cov: float = 1e-6):
        # Epsilon para regularização da covariância
        self.reg_cov = reg_cov
        
        # Atributos que serão definidos no 'fit'
        self.classes_: np.ndarray | None = None
        self.log_priors_: Dict[int, float] = {}
        self.media_: Dict[int, np.ndarray] = {}
        
        # Parâmetros pré-calculados
        self.inv_cov_: Dict[int, np.ndarray] = {}
        self.log_det_cov_: Dict[int, float] = {}
        
        # Constante d * log(2*pi)
        self.log_2pi_d_: float = 0.0
        
        # Atributos que serão definidos pelo decorador @medir_tempo
        self.tempo_treino_ = 0.0
        self.tempo_teste_ = 0.0

    @medir_tempo
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o classificador, calculando médias, matrizes de covariância
        (e suas inversas/determinantes) e priors para cada classe.
        """
        n_amostras, n_features = X.shape
        self.classes_ = np.unique(y)
        
        # Pré-calcula a constante da fórmula da PDF
        self.log_2pi_d_ = n_features * np.log(2 * np.pi)

        # Calcula parâmetros para cada classe
        for classe in self.classes_:
            # 1. Filtra os dados (X) que pertencem a esta classe
            X_classe = X[y == classe]
            
            # 2. Calcula log-prior
            self.log_priors_[classe] = np.log(X_classe.shape[0] / n_amostras)
            
            # 3. Calcula média (vetor)
            self.media_[classe] = np.mean(X_classe, axis=0)
            
            # 4. Calcula Matriz de Covariância
            #    rowvar=False -> assume que colunas são atributos (features)
            
            # -----------------------------------------------------------------
            # CORREÇÃO (Etapa 5 - Debug):
            # Adicionamos ddof=0 (Delta Degrees of Freedom)
            # Isso força np.cov a calcular a variância populacional (dividir por N),
            # que é a estimativa de máxima verossimilhança (MLE),
            # em vez da variância amostral (dividir por N-1).
            # Isso também alinha o cálculo com o nosso teste em utils.py.
            # -----------------------------------------------------------------
            covariancia = np.cov(X_classe, rowvar=False, ddof=0)
            
            # 5. Aplicar Regularização (Σ + εI)
            #    Adiciona 'reg_cov' à diagonal principal
            cov_reg = covariancia + np.eye(n_features) * self.reg_cov
            
            # 6. Pré-calcular inversa e log-determinante (para eficiência)
            #    Usamos 'slogdet' para estabilidade numérica
            sign_det, log_det = np.linalg.slogdet(cov_reg)
            
            # Armazena os parâmetros
            # Verificamos se o determinante é positivo
            if sign_det > 0:
                self.inv_cov_[classe] = np.linalg.inv(cov_reg)
                self.log_det_cov_[classe] = log_det
            else:
                # Se a regularização falhar (raro), atribuímos valores
                # que farão esta classe ter uma probabilidade muito baixa.
                print(f"Aviso: Matriz de covariância da classe {classe} é singular.")
                self.inv_cov_[classe] = np.eye(n_features) # Identidade
                self.log_det_cov_[classe] = np.inf # Log(det) infinito

    def _calcular_log_verossimilhanca(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula o log da verossimilhança (log-likelihood) de cada amostra
        em X para cada classe, usando a fórmula da PDF Gaussiana Multivariada.
        
        Log(PDF) = -0.5 * (d*log(2pi) + log_det_cov + (x-μ).T @ inv_cov @ (x-μ))
        
        Args:
            X (np.ndarray): (n_amostras_teste, n_features)

        Returns:
            np.ndarray: Matriz (n_amostras_teste, n_classes)
        """
        n_amostras = X.shape[0]
        n_classes = len(self.classes_)
        
        # Matriz de log-verossimilhança (amostras x classes)
        log_likelihoods = np.zeros((n_amostras, n_classes))
        
        # Mapeia o valor da classe (ex: 0, 1) para o índice da coluna (ex: 0, 1)
        mapa_classes = {classe: i for i, classe in enumerate(self.classes_)}

        for classe in self.classes_:
            idx_classe = mapa_classes[classe]
            
            # Pega os parâmetros pré-calculados
            media = self.media_[classe]
            inv_cov = self.inv_cov_[classe]
            log_det_cov = self.log_det_cov_[classe]
            
            # 1. Termo constante (escalar)
            constante = self.log_2pi_d_ + log_det_cov
            
            # 2. Termo da Distância de Mahalanobis (vetorizado)
            #    (X - μ)
            diff = X - media # Shape (n_amostras, n_features)
            
            #    (X - μ) @ InvCov
            temp = diff @ inv_cov # Shape (n_amostras, n_features)
            
            #    (X - μ) @ InvCov @ (X - μ).T -> Pegar a diagonal
            #    Forma rápida: sum(A * B) (element-wise)
            dist_mahalanobis_sq = np.sum(temp * diff, axis=1) # Shape (n_amostras,)
            
            # 3. Calcula Log(PDF) e armazena na coluna da classe
            log_likelihoods[:, idx_classe] = -0.5 * (constante + dist_mahalanobis_sq)
            
        return log_likelihoods

    @medir_tempo
    def predict(self, X_teste: np.ndarray) -> np.ndarray:
        """
        Prevê os rótulos para um novo conjunto de dados X_teste.
        """
        if not self.media_:
            raise RuntimeError("O modelo Bayesiano deve ser treinado (fit) antes de prever (predict).")
            
        # 1. Calcula a verossimilhança (likelihood) de X_teste
        #    Shape: (n_amostras, n_classes)
        log_verossimilhanca = self._calcular_log_verossimilhanca(X_teste)
        
        # 2. Calcula a probabilidade posterior (não normalizada)
        #    log(Posterior) = log(Likelihood) + log(Prior)
        
        # Converte o dicionário de priors para um array na ordem correta
        log_priors_array = np.array([self.log_priors_[c] for c in self.classes_])
        
        # Broadcasting: (n_amostras, n_classes) + (1, n_classes)
        log_posterior = log_verossimilhanca + log_priors_array[np.newaxis, :]
        
        # 3. Pega o índice (0 ou 1) da classe com maior log_posterior
        #    axis=1 -> pega o máximo em cada linha (amostra)
        indices_preditos = np.argmax(log_posterior, axis=1)
        
        # 4. Mapeia os índices de volta para os rótulos originais
        #    (ex: índice 0 -> classe 0, índice 1 -> classe 1)
        return self.classes_[indices_preditos]