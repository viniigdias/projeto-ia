"""
Módulo para medição de tempo de execução (treino/teste).

Usamos um decorador que pode ser aplicado a qualquer
método 'fit' ou 'predict' de uma classe.
"""

import time
from functools import wraps

def medir_tempo(func):
    """
    Decorador (@medir_tempo) para medir o tempo de 'fit' e 'predict'.
    
    Ele armazena o tempo gasto (em segundos) em um atributo
    da própria instância do objeto (ex: self.tempo_treino_).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # O 'self' da classe é o primeiro argumento
        instancia = args[0]
        nome_metodo = func.__name__ # Será 'fit' ou 'predict'
        
        t_inicio = time.perf_counter()
        
        # Executa o método original (fit ou predict)
        resultado = func(*args, **kwargs)
        
        t_fim = time.perf_counter()
        tempo_total = t_fim - t_inicio
        
        # Armazena o tempo na instância do objeto
        if nome_metodo == 'fit':
            # Cria o atributo 'tempo_treino_' na classe
            instancia.tempo_treino_ = tempo_total
        elif nome_metodo == 'predict':
            # Cria o atributo 'tempo_teste_' na classe
            instancia.tempo_teste_ = tempo_total
            
        # print(f"[Timing] {instancia.__class__.__name__}.{nome_metodo} levou {tempo_total:.6f}s")
        
        return resultado
    return wrapper