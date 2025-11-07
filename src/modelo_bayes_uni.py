"""
Módulo que implementa o classificador Naive Bayes Gaussiano (Univariado).

Assume que cada atributo (feature) de cada classe segue uma
distribuição Gaussiana independente.
"""

import numpy as np
from timing import medir_tempo

class BayesianoUnivariado:
    """
    Implementação do classificador Naive Bayes Gaussiano (Univariado) do zero.
    
    Parâmetros:
    -----------
    var_epsilon (float): 
        Uma pequena suavização (smoothing) adicionada à variância de cada
        atributo/classe para garantir estabilidade numérica (evitar divisão
        por zero ou log(0) caso um atributo tenha variância zero no treino).
        Conforme solicitado: "suavização σ = max(σ, 1e-6)"
        
    Atributos (pós-fit):
    --------------------
    classes_ (np.ndarray): 
        Array com os rótulos das classes (ex: [0, 1]).
    log_priors_ (np.ndarray): 
        Array (n_classes,) com o logaritmo da probabilidade a priori de cada classe.
    media_ (np.ndarray): 
        Array (n_classes, n_features) com a média de cada atributo para cada classe.
    variancia_ (np.ndarray): 
        Array (n_classes, n_features) com a variância de cada atributo para cada classe.
    tempo_treino_ (float): Tempo (s) do fit, injetado pelo @medir_tempo.
    tempo_teste_ (float): Tempo (s) do predict, injetado pelo @medir_tempo.
    """
    
    def __init__(self, var_epsilon: float = 1e-9):
        # Epsilon para estabilidade numérica (evita variância zero)
        # 1e-9 é um valor padrão (similar ao do sklearn)
        self.var_epsilon = var_epsilon 
        
        # Atributos que serão definidos no 'fit'
        self.classes_: np.ndarray | None = None
        self.log_priors_: np.ndarray | None = None
        self.media_: np.ndarray | None = None
        self.variancia_: np.ndarray | None = None
        
        # Atributos que serão definidos pelo decorador @medir_tempo
        self.tempo_treino_ = 0.0
        self.tempo_teste_ = 0.0

    @medir_tempo
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o classificador, calculando médias, variâncias e priors.
        """
        n_amostras, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Inicializa as matrizes de parâmetros
        self.media_ = np.zeros((n_classes, n_features))
        self.variancia_ = np.zeros((n_classes, n_features))
        self.log_priors_ = np.zeros(n_classes)
        
        # Mapeia o valor da classe (ex: 0, 1) para o índice (ex: 0, 1)
        mapa_classes = {classe: i for i, classe in enumerate(self.classes_)}

        # Calcula parâmetros para cada classe
        for classe in self.classes_:
            idx_classe = mapa_classes[classe]
            
            # 1. Filtra os dados (X) que pertencem a esta classe
            X_classe = X[y == classe]
            
            # 2. Calcula média e variância (por coluna/feature)
            self.media_[idx_classe, :] = np.mean(X_classe, axis=0)
            self.variancia_[idx_classe, :] = np.var(X_classe, axis=0) + self.var_epsilon
            
            # 3. Calcula o log-prior
            # log(P(Classe)) = log(N_amostras_classe / N_total_amostras)
            self.log_priors_[idx_classe] = np.log(X_classe.shape[0] / n_amostras)

    def _calcular_log_verossimilhanca(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula o log da verossimilhança (log-likelihood) de cada amostra
        em X para cada classe, usando a fórmula da PDF Gaussiana.
        
        Log(PDF) = -0.5 * log(2*pi*Var) - ((X - Mu)^2 / (2*Var))
        
        Args:
            X (np.ndarray): (n_amostras_teste, n_features)

        Returns:
            np.ndarray: Matriz (n_amostras_teste, n_classes)
        """
        
        # --- Cálculo Vetorizado (Broadcasting) ---
        
        # X -> (n_amostras, 1, n_features)
        X_b = X[:, np.newaxis, :]
        
        # media_, variancia_ -> (1, n_classes, n_features)
        media_b = self.media_[np.newaxis, :, :]
        var_b = self.variancia_[np.newaxis, :, :]

        # 1. Constante da PDF: -0.5 * log(2 * pi * Var)
        #    (calculado para cada feature/classe)
        constante = -0.5 * np.log(2 * np.pi * var_b)
        
        # 2. Termo do expoente: - (X - Mu)^2 / (2 * Var)
        #    (calculado para cada amostra/feature/classe)
        expoente = -((X_b - media_b) ** 2) / (2 * var_b)
        
        # 3. Soma os dois termos
        #    Shape: (n_amostras, n_classes, n_features)
        log_pdf_por_feature = constante + expoente
        
        # 4. Soma ao longo dos atributos (features) - (a suposição "Naive")
        #    Soma (axis=2) -> Shape: (n_amostras, n_classes)
        log_verossimilhanca_total = np.sum(log_pdf_por_feature, axis=2)
        
        return log_verossimilhanca_total

    @medir_tempo
    def predict(self, X_teste: np.ndarray) -> np.ndarray:
        """
        Prevê os rótulos para um novo conjunto de dados X_teste.
        """
        if self.media_ is None:
            raise RuntimeError("O modelo Bayesiano deve ser treinado (fit) antes de prever (predict).")
            
        # 1. Calcula a verossimilhança (likelihood) de X_teste
        #    Shape: (n_amostras, n_classes)
        log_verossimilhanca = self._calcular_log_verossimilhanca(X_teste)
        
        # 2. Calcula a probabilidade posterior (não normalizada)
        #    log(Posterior) = log(Likelihood) + log(Prior)
        #    Broadcasting: (n_amostras, n_classes) + (1, n_classes)
        log_posterior = log_verossimilhanca + self.log_priors_[np.newaxis, :]
        
        # 3. Pega o índice (0 ou 1) da classe com maior log_posterior
        #    axis=1 -> pega o máximo em cada linha (amostra)
        indices_preditos = np.argmax(log_posterior, axis=1)
        
        # 4. Mapeia os índices de volta para os rótulos originais
        #    (ex: índice 0 -> classe 0, índice 1 -> classe 1)
        return self.classes_[indices_preditos]