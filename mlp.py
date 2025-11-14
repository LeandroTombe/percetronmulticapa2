"""
========================================================================
MLP - Perceptrón Multicapa para Clasificación de Letras
========================================================================

Implementación de una red neuronal feedforward (MLP) para clasificar
patrones de letras distorsionadas (B, D, F).

Características principales:
============================
- Entrada: Matrices 10x10 aplanadas (100 valores binarios)
- Salida: 3 clases (B, D, F) con codificación one-hot
- Algoritmo: Backpropagation + Gradiente descendente con momento
- Inicialización: Xavier/Glorot para convergencia rápida
- Validación: Opcional para detectar overfitting

Arquitectura:
============
Entrada(100) → Oculta1(5-10)[lineal] → [Oculta2(5-10)][lineal] → Salida(3)[sigmoidal]

Algoritmo de entrenamiento:
===========================
1. Forward propagation: calcular salidas capa por capa
2. Calcular error: MSE entre predicción y etiqueta real
3. Backward propagation: calcular gradientes con regla de la cadena
4. Gradiente descendente: actualizar pesos con momento

Fórmulas clave:
==============
- Forward: a_L = f(W_L · a_(L-1) + b_L)
- Error: E = (1/m) · Σ(ŷ - y)²
- Backprop: δ_L = (W^T · δ_(L+1)) · f'(a_L)
- Update: W += -η·∇W + α·ΔW_prev

Autor: Inteligencia Artesanal - Clasificación de Letras
Fecha: 2025
========================================================================
"""
import numpy as np
from typing import List, Callable, Tuple

class MLP:
    """
    Perceptrón Multicapa (Multi-Layer Perceptron) para clasificación de letras.
    
    Implementa una red neuronal feedforward con:
    - Propagación hacia adelante (forward propagation)
    - Retropropagación del error (backpropagation)
    - Gradiente descendente con momento
    
    Arquitectura específica del proyecto:
    =====================================
    - Capa de entrada: 100 neuronas (matriz 10x10 aplanada)
    - Capas ocultas: 1 o 2 capas con 5-10 neuronas c/u (activación LINEAL)
    - Capa de salida: 3 neuronas (activación SIGMOIDAL)
    
    Funciones de activación:
    ========================
    - Capas ocultas: f(z) = z (lineal/identidad)
    - Capa de salida: σ(z) = 1/(1+e^(-z)) (sigmoidal)
    
    Clases de salida:
    ================
    - [1, 0, 0] → Letra B
    - [0, 1, 0] → Letra D
    - [0, 0, 1] → Letra F
    
    Características:
    ===============
    - Inicialización Xavier para convergencia rápida
    - Momento para acelerar entrenamiento
    - Validación opcional para detectar overfitting
    - 3 APIs diferentes para crear la red (retrocompatible)
    """
    
    def __init__(self, capas_ocultas=None, cantidad_neuronas=None, 
                 cantidad_neuronas1=None, cantidad_neuronas2=None,
                 learning_rate: float = 0.1, momentum: float = 0.0,
                 epochs: int = 100):
        """
        Constructor del MLP - Inicializa arquitectura y parámetros.
        
        Funciones de activación (FIJAS):
        ================================
        - Capas ocultas: LINEAL f(z) = z
        - Capa de salida: SIGMOIDAL σ(z) = 1/(1+e^(-z))
        
        Arquitectura resultante:
        =======================
        Entrada(100) → Oculta1(5-10) → [Oculta2(5-10)] → Salida(3)
        
        3 FORMAS DE USO:
        ===============
        
        OPCIÓN 1 - API Clásica (retrocompatible):
        -----------------------------------------
        capas_ocultas : list[int]
            Lista con neuronas por capa oculta
            Ejemplos: [8] para 1 capa, [8, 6] para 2 capas
            
        Ejemplo:
            mlp = MLP(capas_ocultas=[8], learning_rate=0.1, momentum=0.5)
            # Arquitectura: [100, 8, 3]
        
        OPCIÓN 2 - API Nueva (1 capa oculta):
        -------------------------------------
        capas_ocultas : int = 1
            Número de capas ocultas (debe ser 1)
        cantidad_neuronas : int (5-10)
            Cantidad de neuronas en la capa oculta
            
        Ejemplo:
            mlp = MLP(capas_ocultas=1, cantidad_neuronas=8, 
                     learning_rate=0.4, momentum=0.6)
            # Arquitectura: [100, 8, 3]
        
        OPCIÓN 3 - API Nueva (2 capas ocultas):
        ---------------------------------------
        capas_ocultas : int = 2
            Número de capas ocultas (debe ser 2)
        cantidad_neuronas1 : int (5-10)
            Neuronas en primera capa oculta
        cantidad_neuronas2 : int (5-10)
            Neuronas en segunda capa oculta
            
        Ejemplo:
            mlp = MLP(capas_ocultas=2, cantidad_neuronas1=10, 
                     cantidad_neuronas2=6, learning_rate=0.2, momentum=0.3)
            # Arquitectura: [100, 10, 6, 3]
        
        PARÁMETROS COMUNES:
        ==================
        learning_rate : float (0.0 - 1.0)
            Tasa de aprendizaje - controla velocidad de actualización
            Valores típicos: 0.01 (lento) a 0.5 (rápido)
            Default: 0.1
            
        momentum : float (0.0 - 1.0)
            Término de inercia - acelera convergencia
            0.0 = sin momento, 0.9 = alta inercia
            Default: 0.0
            
        epochs : int
            Épocas de entrenamiento por defecto
            Se puede sobrescribir en entrenar()
            Default: 100
        
        Attributes creados:
        ==================
        - arquitectura: [100, ...capas_ocultas..., 3]
        - pesos: Matrices W inicializadas con Xavier
        - sesgos: Vectores b inicializados en 0
        - delta_pesos_anterior: Para momento (inicialmente 0)
        - delta_sesgos_anterior: Para momento (inicialmente 0)
        - activaciones: Lista vacía (se llena en forward)
        - z_values: Lista vacía (se llena en forward)
        """
        # ============================================================
        # PASO 1: Determinar arquitectura según API usada
        # ============================================================
        if isinstance(capas_ocultas, list):
            # API CLÁSICA: capas_ocultas=[8] o [8, 6]
            # Usuario especifica directamente la lista de neuronas
            self.capas_ocultas = capas_ocultas
            
        elif isinstance(capas_ocultas, int):
            # API NUEVA: capas_ocultas=1 o capas_ocultas=2
            # Usuario especifica cantidad de capas y neuronas separadamente
            if capas_ocultas == 1:
                # 1 capa oculta
                if cantidad_neuronas is None:
                    raise ValueError("Para 1 capa oculta, debes especificar 'cantidad_neuronas'")
                self.capas_ocultas = [cantidad_neuronas]
                
            elif capas_ocultas == 2:
                # 2 capas ocultas
                if cantidad_neuronas1 is None or cantidad_neuronas2 is None:
                    raise ValueError("Para 2 capas ocultas, debes especificar 'cantidad_neuronas1' y 'cantidad_neuronas2'")
                self.capas_ocultas = [cantidad_neuronas1, cantidad_neuronas2]
                
            else:
                raise ValueError("capas_ocultas debe ser 1 o 2")
        else:
            raise ValueError("capas_ocultas debe ser una lista [5-10] o un entero (1 o 2)")
        
        # ============================================================
        # PASO 2: Construir arquitectura completa
        # ============================================================
        # Formato: [entrada, oculta1, oculta2?, salida]
        # Ejemplo: [100, 8, 3] o [100, 8, 6, 3]
        self.arquitectura = [100] + self.capas_ocultas + [3]
        self.num_capas = len(self.arquitectura)  # Total de capas (incluyendo entrada)
        
        # ============================================================
        # PASO 3: Guardar hiperparámetros
        # ============================================================
        self.learning_rate = learning_rate  # Tasa de aprendizaje (η)
        self.momentum = momentum            # Término de inercia (α)
        self.epochs = epochs                # Épocas por defecto
        
        # ============================================================
        # PASO 4: Validar parámetros y arquitectura
        # ============================================================
        self._validar_parametros()     # Verifica que η y α estén en [0, 1]
        self._validar_arquitectura()   # Verifica capas y neuronas
        
        # ============================================================
        # PASO 5: Inicializar pesos y sesgos
        # ============================================================
        self.pesos = []   # Lista de matrices W
        self.sesgos = []  # Lista de vectores b
        self._inicializar_pesos()  # Xavier/Glorot initialization
        
        # ============================================================
        # PASO 6: Inicializar variables para momento
        # ============================================================
        # Guardan los ΔW y Δb de la iteración anterior
        # Inicialmente son 0 (sin momento en primera época)
        self.delta_pesos_anterior = [np.zeros_like(w) for w in self.pesos]
        self.delta_sesgos_anterior = [np.zeros_like(b) for b in self.sesgos]
        
        # ============================================================
        # PASO 7: Inicializar variables para propagación
        # ============================================================
        # Se llenarán en cada llamada a forward_propagation
        self.activaciones = []  # Lista de activaciones a_L por capa
        self.z_values = []      # Lista de sumas ponderadas z_L por capa
        
        print(f"✅ MLP creado con arquitectura: {self.arquitectura}")
        print(f"   Capas ocultas: {self.capas_ocultas} (activación LINEAL)")
        print(f"   Capa de salida: 3 neuronas (activación SIGMOIDAL)")
        print(f"   Learning rate: {self.learning_rate}, Momentum: {self.momentum}, Épocas: {self.epochs}")
    
    def _validar_parametros(self):
        """
        Valida que los hiperparámetros estén en rangos válidos.
        
        Verifica que:
        - learning_rate esté entre 0 y 1 (0% a 100% del gradiente)
        - momentum esté entre 0 y 1 (0% a 100% de inercia)
        
        Raises:
            ValueError: Si algún parámetro está fuera del rango válido
        """
        if not (0 <= self.learning_rate <= 1):
            raise ValueError(f"learning_rate debe estar entre 0 y 1 (recibido: {self.learning_rate})")
        
        if not (0 <= self.momentum <= 1):
            raise ValueError(f"momentum debe estar entre 0 y 1 (recibido: {self.momentum})")
    
    def _validar_arquitectura(self):
        """
        Valida que la arquitectura de la red cumpla con los requisitos del proyecto.
        
        Requisitos del TP:
        - Exactamente 1 o 2 capas ocultas
        - Cada capa oculta debe tener entre 5 y 10 neuronas
        - Entrada fija: 100 neuronas (matriz 10x10 aplanada)
        - Salida fija: 3 neuronas (B, D, F)
        
        Raises:
            ValueError: Si la arquitectura no cumple con los requisitos
        """
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
        Inicializa los pesos y sesgos de la red neuronal.
        
        Usa inicialización Xavier/Glorot:
        - Pesos: valores aleatorios uniformes en [-límite, +límite]
        - límite = sqrt(6 / (neuronas_entrada + neuronas_salida))
        - Sesgos: inicializados en 0
        
        Ventajas de Xavier:
        - Evita que las activaciones exploten o se desvanezcan
        - Mantiene varianza similar entre capas
        - Mejora la convergencia del entrenamiento
        
        Nota: Usa semilla fija (42) para reproducibilidad de resultados
        """
        np.random.seed(42)  # Para reproducibilidad
        
        for i in range(self.num_capas - 1):
            # Número de neuronas de entrada y salida para esta capa
            n_entrada = self.arquitectura[i]
            n_salida = self.arquitectura[i + 1]
            
            # Inicialización Xavier: límite = sqrt(6 / (n_in + n_out))
            limite = np.sqrt(6 / (n_entrada + n_salida))
            w = np.random.uniform(-limite, limite, (n_entrada, n_salida))
            b = np.zeros((1, n_salida))
            
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def _sigmoidal(self, z: np.ndarray) -> np.ndarray:
        """
        Función de activación sigmoidal (logística).
        
        Fórmula: σ(z) = 1 / (1 + e^(-z))
        Rango de salida: (0, 1)
        
        Uso en este MLP: Capa de salida (clasificación de 3 clases)
        
        Args:
            z: Valor de entrada (suma ponderada + sesgo)
            
        Returns:
            Activación en rango (0, 1)
            
        Nota: Aplica clip a z para evitar overflow en exp()
        """
        # Clip para evitar overflow en valores muy grandes/pequeños
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _sigmoidal_derivada(self, a: np.ndarray) -> np.ndarray:
        """
        Derivada de la función sigmoidal.
        
        Fórmula: σ'(a) = a * (1 - a)
        donde 'a' es la activación ya calculada (no z)
        
        Uso: Backpropagation en capa de salida
        
        Args:
            a: Activación de la neurona (salida de sigmoidal)
            
        Returns:
            Derivada evaluada en 'a'
        """
        return a * (1 - a)
    
    def _lineal(self, z: np.ndarray) -> np.ndarray:
        """
        Función de activación lineal (identidad).
        
        Fórmula: f(z) = z
        Rango de salida: (-∞, +∞)
        
        Uso en este MLP: Capas ocultas
        
        Args:
            z: Valor de entrada
            
        Returns:
            El mismo valor de entrada (sin transformación)
        """
        return z
    
    def _lineal_derivada(self, a: np.ndarray) -> np.ndarray:
        """
        Derivada de la función lineal.
        
        Fórmula: f'(a) = 1
        
        Uso: Backpropagation en capas ocultas
        
        Args:
            a: Activación de la neurona (no se usa, pero se mantiene
               por consistencia con otras derivadas)
            
        Returns:
            Array de unos con la misma forma que 'a'
        """
        return np.ones_like(a)
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Propaga la entrada hacia adelante a través de toda la red (feedforward).
        
        Proceso por capa:
        1. Calcula z = W·a + b (suma ponderada + sesgo)
        2. Aplica función de activación: a = f(z)
        3. Guarda z y a para usar en backpropagation
        
        Funciones de activación:
        - Capas ocultas: LINEAL (f(z) = z)
        - Capa de salida: SIGMOIDAL (σ(z) = 1/(1+e^(-z)))
        
        Ejemplo de flujo:
        X (100) → [W1]→ z1 → f(z1) → a1 (8) → [W2]→ z2 → σ(z2) → salida (3)
        
        Args:
            X: Datos de entrada, shape (n_muestras, 100)
               Cada fila es un patrón de letra de 10x10 aplanado
        
        Returns:
            Salida de la red, shape (n_muestras, 3)
            Probabilidades para cada clase (B, D, F)
            
        Nota: Guarda activaciones y z_values en self para backpropagation
        """
        # Inicializar listas para guardar valores intermedios
        self.activaciones = [X]  # a0 = entrada
        self.z_values = []       # z = W·a + b de cada capa
        
        a = X  # Activación inicial = entrada
        
        # Iterar por cada capa
        for i in range(self.num_capas - 1):
            # 1. Calcular suma ponderada: z = W·a + b
            z = np.dot(a, self.pesos[i]) + self.sesgos[i]
            self.z_values.append(z)
            
            # 2. Aplicar función de activación según tipo de capa
            if i == self.num_capas - 2:  # Última capa (salida)
                a = self._sigmoidal(z)    # Sigmoidal para probabilidades
            else:  # Capas ocultas
                a = self._lineal(z)       # Lineal para capas internas
            
            # 3. Guardar activación para backpropagation
            self.activaciones.append(a)
        
        return a  # Retorna salida de la última capa
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Calcula los errores (deltas) de cada capa mediante retropropagación.
        
        Algoritmo de backpropagation:
        1. Calcula error de salida: δ_salida = (ŷ - y) · σ'(ŷ)
        2. Propaga error hacia atrás: δ_L = W^T · δ_(L+1) · f'(a_L)
        3. Repite para cada capa hasta llegar a la entrada
        
        Fórmulas por capa:
        - Capa de salida: δ = (y_pred - y_real) * σ'(y_pred)
        - Capas ocultas: δ = (W^T · δ_siguiente) * f'(a)
        
        Donde:
        - y_pred: salida obtenida de la red
        - y_real: salida esperada (etiqueta)
        - σ': derivada de sigmoidal
        - f': derivada de lineal (= 1)
        - W^T: matriz de pesos transpuesta
        
        Args:
            X: Datos de entrada (no se usan, pero se mantienen por
               compatibilidad con la API estándar)
            y: Etiquetas verdaderas, shape (n_muestras, 3)
               One-hot encoding de las clases B, D, F
        
        Returns:
            Lista de deltas para cada capa [δ0, δ1, ..., δ_salida]
            Cada delta tiene la forma de las activaciones de esa capa
            
        Nota: Usa las activaciones guardadas en forward_propagation
        """
        # 1. Calcular delta de capa de salida
        # δ_salida = (ŷ - y) * σ'(ŷ)
        y_obtenido = self.activaciones[-1]  # Predicción de la red
        delta_salida = (y_obtenido - y) * self._sigmoidal_derivada(y_obtenido)
        
        # Inicializar lista con delta de salida
        deltas = [delta_salida]
        
        # 2. Retropropagar el error desde la salida hacia la entrada
        for i in reversed(range(len(self.pesos))):
            # δ_L = (W^T · δ_(L+1)) * f'(a_L)
            # W^T: transpuesta de matriz de pesos
            # δ_(L+1): error de la capa siguiente
            # f'(a_L): derivada de activación lineal (= 1)
            delta = np.dot(deltas[0], self.pesos[i].T) * self._lineal_derivada(self.activaciones[i])
            deltas.insert(0, delta)  # Insertar al inicio para mantener orden
        
        # 3. Retornar deltas en orden: [δ_entrada, δ_oculta1, ..., δ_salida]
        return deltas
    
    def gradiente_descendente(self, deltas: List[np.ndarray]):
        """
        Actualiza pesos y sesgos usando gradiente descendente con momento.
        
        Algoritmo:
        1. Calcula gradientes: ∇W = (a^T · δ) / m
        2. Aplica momento: ΔW = -η·∇W + α·ΔW_anterior
        3. Actualiza pesos: W_nuevo = W_viejo + ΔW
        
        Fórmulas:
        - Gradiente de pesos: ∇W = (1/m) · a^T · δ
        - Gradiente de sesgos: ∇b = (1/m) · Σ(δ)
        - Delta con momento: ΔW = -η·∇W + α·ΔW_prev
        - Actualización: W += ΔW, b += Δb
        
        Donde:
        - m: tamaño del batch (número de ejemplos)
        - a: activaciones de la capa anterior
        - δ: error de la capa actual
        - η: learning_rate (tasa de aprendizaje)
        - α: momentum (término de inercia)
        
        Ventajas del momento:
        - Acelera convergencia en direcciones consistentes
        - Suaviza oscilaciones en el descenso
        - Ayuda a escapar de mínimos locales poco profundos
        
        Args:
            deltas: Lista de errores por capa calculados en backpropagation
                   [δ0, δ1, ..., δ_salida]
                   
        Nota: Usa operaciones vectorizadas de NumPy para eficiencia
        """
        m = self.activaciones[0].shape[0]  # Batch size para normalización
        
        # Actualizar pesos y sesgos para cada capa
        for i in range(len(self.pesos)):
            # 1. Calcular gradientes usando operaciones vectorizadas
            # ∇W = (1/m) · a^T · δ
            grad_w = np.dot(self.activaciones[i].T, deltas[i + 1]) / m
            # ∇b = (1/m) · Σ(δ) sobre todas las muestras
            grad_b = np.sum(deltas[i + 1], axis=0, keepdims=True) / m
            
            # 2. Aplicar momento: ΔW = -η·∇W + α·ΔW_anterior
            # Combina gradiente actual con dirección previa (inercia)
            delta_w = -self.learning_rate * grad_w + self.momentum * self.delta_pesos_anterior[i]
            delta_b = -self.learning_rate * grad_b + self.momentum * self.delta_sesgos_anterior[i]
            
            # 3. Actualizar pesos y sesgos
            self.pesos[i] += delta_w
            self.sesgos[i] += delta_b
            
            # 4. Guardar deltas actuales para usar en próxima iteración (momento)
            self.delta_pesos_anterior[i] = delta_w
            self.delta_sesgos_anterior[i] = delta_b
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray = None, y_val: np.ndarray = None,
                epochs: int = None, verbose: bool = True):
        """
        Entrena la red neuronal con los datos de entrenamiento.
        
        Proceso de entrenamiento por época:
        1. Forward propagation: calcular predicciones
        2. Calcular error MSE (Mean Squared Error)
        3. Backward propagation: calcular deltas de error
        4. Gradiente descendente: actualizar pesos y sesgos
        5. (Opcional) Evaluar en conjunto de validación
        
        Tipos de entrenamiento:
        - Sin validación: Solo entrena y reporta error de entrenamiento
        - Con validación: Entrena y evalúa en datos no vistos (detecta overfitting)
        
        Función de error (MSE):
        MSE = (1/m) · Σ(y_pred - y_real)²
        Donde m = número de ejemplos
        
        Args:
            X_train: Datos de entrenamiento, shape (n_ejemplos, 100)
            y_train: Etiquetas de entrenamiento, shape (n_ejemplos, 3)
                    One-hot encoding: [1,0,0]=B, [0,1,0]=D, [0,0,1]=F
            X_val: (Opcional) Datos de validación, shape (n_val, 100)
            y_val: (Opcional) Etiquetas de validación, shape (n_val, 3)
            epochs: Número de épocas (iteraciones completas sobre el dataset)
                   Si es None, usa self.epochs definido en __init__
            verbose: Si True, imprime progreso época por época
        
        Returns:
            Si hay validación: dict con {'train_loss': [...], 'val_loss': [...]}
            Si no hay validación: list con historial de MSE de entrenamiento
            
        Nota: El error de validación se calcula DESPUÉS de actualizar pesos,
              por lo que refleja el rendimiento del modelo actualizado
        """
        # Si no se especifican épocas, usar las del constructor
        if epochs is None:
            epochs = self.epochs
        
        # Inicializar historiales de error
        historial_train = []
        historial_val = []
        usar_validacion = X_val is not None and y_val is not None
        
        # Ciclo de entrenamiento
        for epoch in range(epochs):
            # 1. Forward propagation: calcular predicciones
            y_pred = self.forward_propagation(X_train)
            
            # 2. Calcular error de entrenamiento (MSE)
            # MSE = promedio de (predicción - real)²
            error_train = np.mean((y_train - y_pred) ** 2)
            historial_train.append(error_train)
            
            # 3. Backward propagation: calcular deltas de error
            # IMPORTANTE: Hacer ANTES de validación para no sobrescribir activaciones
            deltas = self.backward_propagation(X_train, y_train)
            
            # 4. Gradiente descendente: actualizar pesos y sesgos
            self.gradiente_descendente(deltas)
            
            # 5. Calcular error de validación (si se proporcionó)
            # Se hace DESPUÉS de actualizar pesos
            if usar_validacion:
                y_val_pred = self.forward_propagation(X_val)
                error_val = np.mean((y_val - y_val_pred) ** 2)
                historial_val.append(error_val)
            
            # 6. Mostrar progreso (si verbose=True)
            if verbose:
                if usar_validacion:
                    print(f"Época {epoch + 1}/{epochs} - "
                          f"Error Entrenamiento (MSE): {error_train:.6f} - "
                          f"Error Validación (MSE): {error_val:.6f}")
                else:
                    print(f"Época {epoch + 1}/{epochs} - Error (MSE): {error_train:.6f}")
        
        # Retornar historial en formato apropiado
        if usar_validacion:
            return {'train_loss': historial_train, 'val_loss': historial_val}
        else:
            return historial_train  # Retrocompatible con código antiguo
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos (inferencia).
        
        Ejecuta forward propagation para obtener las salidas de la red
        sin realizar entrenamiento (no actualiza pesos).
        
        Uso típico:
        - Clasificar nuevos patrones después del entrenamiento
        - Evaluar rendimiento en conjunto de test
        - Obtener probabilidades de cada clase
        
        Args:
            X: Datos de entrada, shape (n_muestras, 100)
               Cada fila es un patrón de letra 10x10 aplanado
        
        Returns:
            Predicciones, shape (n_muestras, 3)
            Cada fila contiene 3 valores en rango [0,1] que suman ~1
            Representan probabilidades de ser B, D o F
            
        Ejemplo:
            pred = mlp.predecir([[0,1,1,...,0]])  # Un patrón
            # pred = [[0.1, 0.8, 0.1]]  → Probablemente es 'D'
            clase = np.argmax(pred)  # 1 → 'D'
        """
        return self.forward_propagation(X)
