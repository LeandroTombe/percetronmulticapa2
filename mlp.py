"""
Implementación de un Perceptrón Multicapa (Multi-Layer Perceptron)
para clasificación de patrones.
"""

import numpy as np
from typing import List, Callable, Tuple


class MLP:
    """
    Clase que implementa un Perceptrón Multicapa (MLP).
    
    Permite definir una arquitectura personalizable con:
    - Número de capas
    - Número de neuronas por capa
    - Funciones de activación por capa
    """
    
    def __init__(self, arquitectura: List[int], funciones_activacion: List[str], 
                 learning_rate: float = 0.1, momentum: float = 0.0):
        """
        Inicializa el MLP con la arquitectura especificada.
        
        Parámetros:
        -----------
        arquitectura : List[int]
            Lista con el número de neuronas por capa.
            Ejemplo: [100, 10, 5, 10] significa:
            - Capa de entrada: 100 neuronas (características)
            - Capa oculta 1: 10 neuronas
            - Capa oculta 2: 5 neuronas
            - Capa de salida: 10 neuronas (clases)
            
        funciones_activacion : List[str]
            Lista con las funciones de activación para cada capa (excepto entrada).
            Opciones: 'lineal', 'sigmoidal'
            Ejemplo: ['sigmoidal', 'sigmoidal', 'lineal']
            
        learning_rate : float
            Coeficiente de aprendizaje (entre 0 y 1)
            
        momentum : float
            Término momento (entre 0 y 1)
        """
        self.arquitectura = arquitectura
        self.num_capas = len(arquitectura)
        self.funciones_activacion = funciones_activacion
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Validaciones
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
        
        print(f"MLP creado con arquitectura: {arquitectura}")
        print(f"Funciones de activación: {funciones_activacion}")
        print(f"Learning rate: {learning_rate}, Momentum: {momentum}")
    
    def _validar_arquitectura(self):
        """Valida que la arquitectura sea correcta según los requisitos."""
        # Debe haber al menos 3 capas (entrada, oculta, salida)
        if self.num_capas < 3:
            raise ValueError("La red debe tener al menos 3 capas (entrada, oculta, salida)")
        
        # Validar número de capas ocultas (1 o 2 según requisitos)
        num_capas_ocultas = self.num_capas - 2
        if num_capas_ocultas < 1 or num_capas_ocultas > 2:
            raise ValueError("Debe haber 1 o 2 capas ocultas")
        
        # Validar número de neuronas por capa (5 a 10 según requisitos)
        for i, num_neuronas in enumerate(self.arquitectura[1:]):  # Excepto entrada
            if num_neuronas < 5 or num_neuronas > 10:
                raise ValueError(f"La capa {i+1} debe tener entre 5 y 10 neuronas (tiene {num_neuronas})")
        
        # Validar funciones de activación
        funciones_validas = ['lineal', 'sigmoidal']
        if len(self.funciones_activacion) != self.num_capas - 1:
            raise ValueError(f"Debe especificar {self.num_capas - 1} funciones de activación")
        
        for func in self.funciones_activacion:
            if func not in funciones_validas:
                raise ValueError(f"Función de activación '{func}' no válida. Use: {funciones_validas}")
    
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
            
            # Aplicar función de activación
            a = self._aplicar_activacion(z, self.funciones_activacion[i])
            self.activaciones.append(a)
        
        return a
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Propaga el error hacia atrás y calcula los gradientes.
        
        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrada
        y : np.ndarray
            Etiquetas verdaderas
        
        Retorna:
        --------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Gradientes para pesos y sesgos
        """
        m = X.shape[0]
        gradientes_pesos = []
        gradientes_sesgos = []
        
        # Error en la capa de salida
        delta = self.activaciones[-1] - y
        
        # Retropropagar el error
        for i in range(self.num_capas - 2, -1, -1):
            # Gradientes
            grad_w = np.dot(self.activaciones[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradientes_pesos.insert(0, grad_w)
            gradientes_sesgos.insert(0, grad_b)
            
            # Propagar el error a la capa anterior (si no es la primera capa)
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T) * \
                        self._aplicar_derivada_activacion(self.activaciones[i], 
                                                         self.funciones_activacion[i-1])
        
        return gradientes_pesos, gradientes_sesgos
    
    def actualizar_pesos(self, gradientes_pesos: List[np.ndarray], 
                        gradientes_sesgos: List[np.ndarray]):
        """
        Actualiza los pesos y sesgos usando los gradientes y el término momento.
        
        Parámetros:
        -----------
        gradientes_pesos : List[np.ndarray]
            Gradientes de los pesos
        gradientes_sesgos : List[np.ndarray]
            Gradientes de los sesgos
        """
        for i in range(len(self.pesos)):
            # Calcular delta con momento
            delta_w = -self.learning_rate * gradientes_pesos[i] + \
                     self.momentum * self.delta_pesos_anterior[i]
            delta_b = -self.learning_rate * gradientes_sesgos[i] + \
                     self.momentum * self.delta_sesgos_anterior[i]
            
            # Actualizar pesos
            self.pesos[i] += delta_w
            self.sesgos[i] += delta_b
            
            # Guardar deltas para el próximo paso
            self.delta_pesos_anterior[i] = delta_w
            self.delta_sesgos_anterior[i] = delta_b
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray, 
                epochs: int = 1000, verbose: bool = True) -> List[float]:
        """
        Entrena el MLP con los datos proporcionados.
        
        Parámetros:
        -----------
        X_train : np.ndarray
            Datos de entrenamiento
        y_train : np.ndarray
            Etiquetas de entrenamiento
        epochs : int
            Número de épocas de entrenamiento
        verbose : bool
            Si True, muestra el progreso del entrenamiento
        
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
            
            # Backward propagation
            gradientes_pesos, gradientes_sesgos = self.backward_propagation(X_train, y_train)
            
            # Actualizar pesos
            self.actualizar_pesos(gradientes_pesos, gradientes_sesgos)
            
            # Mostrar progreso
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Época {epoch}/{epochs} - Error (MSE): {error:.6f}")
        
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
            resumen += f"  - Función de activación: {self.funciones_activacion[i]}\n"
            resumen += f"  - Pesos: {self.pesos[i].shape}\n"
            resumen += f"  - Sesgos: {self.sesgos[i].shape}\n\n"
        
        return resumen


if __name__ == "__main__":
    # Ejemplo de uso
    print("Ejemplo de creación de MLP con arquitectura personalizable\n")
    
    # Ejemplo 1: Red simple (1 capa oculta)
    print("Ejemplo 1: Red con 1 capa oculta")
    mlp1 = MLP(
        arquitectura=[100, 10, 10],  # 100 entradas, 10 ocultas, 10 salidas
        funciones_activacion=['sigmoidal', 'lineal'],
        learning_rate=0.1,
        momentum=0.5
    )
    print(mlp1.obtener_resumen())
    
    # Ejemplo 2: Red con 2 capas ocultas
    print("\nEjemplo 2: Red con 2 capas ocultas")
    mlp2 = MLP(
        arquitectura=[100, 8, 6, 10],  # 100 entradas, 8 ocultas, 6 ocultas, 10 salidas
        funciones_activacion=['sigmoidal', 'sigmoidal', 'lineal'],
        learning_rate=0.05,
        momentum=0.9
    )
    print(mlp2.obtener_resumen())
