"""
Implementación de un Perceptrón Multicapa (Multi-Layer Perceptron)
para clasificación de patrones.
"""

import numpy as np
from typing import List, Callable, Tuple


class MLP:
    """
    Clase que implementa un Perceptrón Multicapa (MLP).
    
    Red específica para clasificación de 3 letras (B, D, F):
    - Entrada fija: 100 neuronas (matriz 10x10)
    - Capas ocultas: 1 o 2 capas con 5-10 neuronas cada una
    - Salida fija: 3 neuronas (una por letra)
    """
    
    def __init__(self, capas_ocultas: List[int], 
                 learning_rate: float = 0.1, momentum: float = 0.0):
        """
        Inicializa el MLP con la arquitectura especificada.
        
        Funcionalidad fija:
        - Capas ocultas: activación LINEAL
        - Capa de salida: activación SIGMOIDAL
        
        Parámetros:
        -----------
        capas_ocultas : List[int]
            Lista con el número de neuronas en cada capa oculta.
            - Para 1 capa oculta: [8] significa 8 neuronas
            - Para 2 capas ocultas: [8, 6] significa 8 neuronas en la primera, 6 en la segunda
            Cada capa debe tener entre 5 y 10 neuronas.
            
        learning_rate : float
            Coeficiente de aprendizaje (entre 0 y 1)
            
        momentum : float
            Término momento (entre 0 y 1)
        
        Ejemplos:
        ---------
        # Red con 1 capa oculta de 8 neuronas
        mlp = MLP(capas_ocultas=[8], learning_rate=0.1, momentum=0.5)
        
        # Red con 2 capas ocultas de 10 y 6 neuronas
        mlp = MLP(capas_ocultas=[10, 6], learning_rate=0.05, momentum=0.9)
        """
        # Construir arquitectura completa: entrada (100) + capas_ocultas + salida (3)
        self.capas_ocultas = capas_ocultas
        self.arquitectura = [100] + capas_ocultas + [3]
        self.num_capas = len(self.arquitectura)
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Validaciones
        self._validar_parametros()
        self._validar_arquitectura()
        
        # Inicializar pesos y sesgos
        self.pesos = []
        self.sesgos = []
        self._inicializar_pesos()
        
        # Para el término momento
        self.delta_pesos_anterior = [np.zeros_like(w) for w in self.pesos]
        self.delta_sesgos_anterior = [np.zeros_like(b) for b in self.sesgos]
        
        # Para almacenar valores durante la propagación
        self.activaciones = []
        self.z_values = []
        
        print(f"✅ MLP creado con arquitectura: {self.arquitectura}")
        print(f"   Capas ocultas: {self.capas_ocultas} (activación LINEAL)")
        print(f"   Capa de salida: 3 neuronas (activación SIGMOIDAL)")
        print(f"   Learning rate: {self.learning_rate}, Momentum: {self.momentum}")
    
    def _validar_parametros(self):
        """Valida los parámetros learning_rate y momentum."""
        if not (0 <= self.learning_rate <= 1):
            raise ValueError(f"learning_rate debe estar entre 0 y 1 (recibido: {self.learning_rate})")
        
        if not (0 <= self.momentum <= 1):
            raise ValueError(f"momentum debe estar entre 0 y 1 (recibido: {self.momentum})")
    
    def _validar_arquitectura(self):
        """Valida que la arquitectura sea correcta según los requisitos."""
        # Validar número de capas ocultas (1 o 2)
        num_capas_ocultas = len(self.capas_ocultas)
        if num_capas_ocultas < 1 or num_capas_ocultas > 2:
            raise ValueError(f"Debe especificar 1 o 2 capas ocultas (recibió {num_capas_ocultas})")

        # Validar número de neuronas por capa OCULTA (5 a 10)
        for i, num_neuronas in enumerate(self.capas_ocultas, 1):
            if num_neuronas < 5 or num_neuronas > 10:
                raise ValueError(f"La capa oculta {i} debe tener entre 5 y 10 neuronas (tiene {num_neuronas})")
    
    def _inicializar_pesos(self):
        """
        Inicializa los pesos y sesgos de la red con valores aleatorios pequeños.
        Usa inicialización Xavier/Glorot para mejor convergencia.
        """
        np.random.seed(42)  # Para reproducibilidad
        
        for i in range(self.num_capas - 1):
            # Número de neuronas de entrada y salida para esta capa
            n_entrada = self.arquitectura[i]
            n_salida = self.arquitectura[i + 1]
            
            # Inicialización Xavier
            limite = np.sqrt(6 / (n_entrada + n_salida))
            w = np.random.uniform(-limite, limite, (n_entrada, n_salida))
            b = np.zeros((1, n_salida))
            
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def _sigmoidal(self, z: np.ndarray) -> np.ndarray:
        """Función de activación sigmoidal (logística)."""
        # Clip para evitar overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _sigmoidal_derivada(self, a: np.ndarray) -> np.ndarray:
        """Derivada de la función sigmoidal."""
        return a * (1 - a)
    
    def _lineal(self, z: np.ndarray) -> np.ndarray:
        """Función de activación lineal (identidad)."""
        return z
    
    def _lineal_derivada(self, a: np.ndarray) -> np.ndarray:
        """Derivada de la función lineal."""
        return np.ones_like(a)
    
    def _aplicar_activacion(self, z: np.ndarray, funcion: str) -> np.ndarray:
        """Aplica la función de activación especificada."""
        if funcion == 'sigmoidal':
            return self._sigmoidal(z)
        elif funcion == 'lineal':
            return self._lineal(z)
        else:
            raise ValueError(f"Función de activación desconocida: {funcion}")
    
    def _aplicar_derivada_activacion(self, a: np.ndarray, funcion: str) -> np.ndarray:
        """Aplica la derivada de la función de activación especificada."""
        if funcion == 'sigmoidal':
            return self._sigmoidal_derivada(a)
        elif funcion == 'lineal':
            return self._lineal_derivada(a)
        else:
            raise ValueError(f"Función de activación desconocida: {funcion}")
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Propaga la entrada hacia adelante a través de la red.
        
        Funcionalidad:
        - Capas ocultas: activación LINEAL
        - Capa de salida: activación SIGMOIDAL
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrada de forma (n_muestras, n_caracteristicas)
        
        Retorna:
        --------
        np.ndarray
            Salida de la red de forma (n_muestras, n_salidas)
        """
        self.activaciones = [X]
        self.z_values = []
        
        a = X
        for i in range(self.num_capas - 1):
            # Calcular z = w*a + b
            z = np.dot(a, self.pesos[i]) + self.sesgos[i]
            self.z_values.append(z)
            
            # Aplicar función de activación según la capa
            # Si es la última capa (salida): sigmoidal
            # Si es capa oculta: lineal
            if i == self.num_capas - 2:  # Última capa (salida)
                a = self._sigmoidal(z)
            else:  # Capas ocultas
                a = self._lineal(z)
            
            self.activaciones.append(a)
        
        return a
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Calcula los deltas de error para cada capa mediante backpropagation.
        
        Similar a la implementación original pero optimizada:
        - Calcula delta de salida con derivada de sigmoidal
        - Propaga hacia atrás multiplicando por derivada lineal
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrada
        y : np.ndarray
            Etiquetas verdaderas (salida esperada)
        
        Retorna:
        --------
        List[np.ndarray]
            Lista de deltas para cada capa (desde entrada hasta salida)
        """
        # Delta de capa de salida: (y_obtenido - y_esperado) * derivada_sigmoidal(y_obtenido)
        y_obtenido = self.activaciones[-1]
        delta_salida = (y_obtenido - y) * self._sigmoidal_derivada(y_obtenido)
        
        # Lista para almacenar deltas
        deltas = [delta_salida]
        
        # Retropropagar el error desde la salida hacia la entrada
        for i in reversed(range(len(self.pesos))):
            # delta = w^T * delta_siguiente * derivada_lineal
            delta = np.dot(deltas[0], self.pesos[i].T) * self._lineal_derivada(self.activaciones[i])
            deltas.insert(0, delta)
        
        # Retornar deltas en orden: [delta_entrada, delta_oculta1, ..., delta_salida]
        return deltas
    
    def gradiente_descendente(self, deltas: List[np.ndarray]):
        """
        Actualiza pesos y sesgos usando gradiente descendente con momento.
        
        Versión optimizada que usa operaciones vectorizadas de NumPy
        en lugar de loops anidados (mucho más rápido).
        
        Parámetros:
        -----------
        deltas : List[np.ndarray]
            Lista de deltas calculados por backpropagation
        """
        m = self.activaciones[0].shape[0]  # Batch size para normalización
        
        # Actualizar pesos y sesgos para cada capa
        for i in range(len(self.pesos)):
            # Calcular gradientes usando operaciones vectorizadas
            # grad_w = (activaciones[i]^T * deltas[i+1]) / m
            grad_w = np.dot(self.activaciones[i].T, deltas[i + 1]) / m
            grad_b = np.sum(deltas[i + 1], axis=0, keepdims=True) / m
            
            # Aplicar momento estándar: delta_w = -lr*grad + momentum*delta_anterior
            delta_w = -self.learning_rate * grad_w + self.momentum * self.delta_pesos_anterior[i]
            delta_b = -self.learning_rate * grad_b + self.momentum * self.delta_sesgos_anterior[i]
            
            # Actualizar pesos
            self.pesos[i] += delta_w
            self.sesgos[i] += delta_b
            
            # Guardar deltas para siguiente iteración (momento)
            self.delta_pesos_anterior[i] = delta_w
            self.delta_sesgos_anterior[i] = delta_b
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray, 
                epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        Entrena el MLP con los datos proporcionados.
        
        Parámetros:
        -----------
        X_train : np.ndarray
            Datos de entrenamiento
        y_train : np.ndarray
            Etiquetas de entrenamiento
        epochs : int
            Número de épocas de entrenamiento (pasado por parámetro)
        verbose : bool
            Si True, muestra el progreso del entrenamiento época por época
        
        Retorna:
        --------
        List[float]
            Historia de los errores por época
        """
        historial_errores = []
        
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X_train)
            
            # Calcular error (MSE)
            error = np.mean((y_train - y_pred) ** 2)
            historial_errores.append(error)
            
            # Backward propagation: calcular deltas
            deltas = self.backward_propagation(X_train, y_train)
            
            # Gradiente descendente: actualizar pesos y sesgos
            self.gradiente_descendente(deltas)
            
            # Mostrar progreso época por época
            if verbose:
                print(f"Época {epoch + 1}/{epochs} - Error (MSE): {error:.6f}")
        
        return historial_errores
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos.
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrada
        
        Retorna:
        --------
        np.ndarray
            Predicciones de la red
        """
        return self.forward_propagation(X)
    
    def obtener_resumen(self) -> str:
        """
        Retorna un resumen de la arquitectura de la red.
        """
        resumen = "=" * 50 + "\n"
        resumen += "RESUMEN DE LA RED NEURONAL MLP\n"
        resumen += "=" * 50 + "\n\n"
        resumen += f"Arquitectura: {self.arquitectura}\n"
        resumen += f"Número total de capas: {self.num_capas}\n"
        resumen += f"Capas ocultas: {self.num_capas - 2}\n"
        resumen += f"Learning rate: {self.learning_rate}\n"
        resumen += f"Momentum: {self.momentum}\n\n"
        
        for i in range(self.num_capas - 1):
            resumen += f"Capa {i} -> Capa {i+1}:\n"
            resumen += f"  - Neuronas: {self.arquitectura[i]} -> {self.arquitectura[i+1]}\n"
            
            # Determinar función de activación según la capa
            if i == self.num_capas - 2:  # Última capa (salida)
                funcion_act = "sigmoidal"
            else:  # Capas ocultas
                funcion_act = "lineal"
            
            resumen += f"  - Función de activación: {funcion_act}\n"
            resumen += f"  - Pesos: {self.pesos[i].shape}\n"
            resumen += f"  - Sesgos: {self.sesgos[i].shape}\n\n"
        
        return resumen


if __name__ == "__main__":
    # Ejemplo de uso
    print("Ejemplo de creación de MLP para clasificación de letras B, D, F\n")
    print("="*60)
    
    # Ejemplo 1: Red simple (1 capa oculta)
    print("\n📌 Ejemplo 1: Red con 1 capa oculta de 8 neuronas")
    print("-"*60)
    mlp1 = MLP(
        capas_ocultas=[8],  # 1 capa oculta con 8 neuronas (activación lineal)
        learning_rate=0.1,
        momentum=0.5
    )
    print(mlp1.obtener_resumen())
    
    # Ejemplo 2: Red con 2 capas ocultas
    print("\n📌 Ejemplo 2: Red con 2 capas ocultas (8 y 6 neuronas)")
    print("-"*60)
    mlp2 = MLP(
        capas_ocultas=[8, 6],  # 2 capas ocultas: 8 y 6 neuronas (activación lineal)
        learning_rate=0.05,
        momentum=0.9
    )
    print(mlp2.obtener_resumen())
    
    # Ejemplo 3: Arquitectura completa resultante
    print("\n📌 Ejemplo 3: Diferentes configuraciones")
    print("-"*60)
    print("capas_ocultas=[5]    → Arquitectura: [100, 5, 3]")
    print("capas_ocultas=[10]   → Arquitectura: [100, 10, 3]")
    print("capas_ocultas=[7, 5] → Arquitectura: [100, 7, 5, 3]")
    print("capas_ocultas=[10,8] → Arquitectura: [100, 10, 8, 3]")
    
    # Ejemplo 4: Probar validaciones
    print("\n📌 Ejemplo 4: Validaciones")
    print("-"*60)
    try:
        mlp_error = MLP(capas_ocultas=[15])
    except ValueError as e:
        print(f"❌ Error esperado: {e}")
    
    try:
        mlp_error2 = MLP(capas_ocultas=[8], learning_rate=1.5)
    except ValueError as e:
        print(f"❌ Error esperado: {e}")
