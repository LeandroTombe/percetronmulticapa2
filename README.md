# Perceptrón Multicapa (MLP) - TP 2025

Implementación de un Perceptrón Multicapa para clasificación de patrones de letras (B, D, F).

## 🚀 Inicio Rápido

### Opción 1: Script Interactivo (Recomendado)

```bash
python entrenar_mlp.py
```

Este script te guiará paso a paso:
1. ✅ **Selecciona cantidad**: 100 / 500 / 1000 ejemplos
2. ✅ **Configura distorsión**: 1-30%
3. ✅ **Define arquitectura**: 1 o 2 capas ocultas
4. ✅ **Ajusta hiperparámetros**: learning rate, momentum, epochs
5. ✅ **Entrena y evalúa** automáticamente

### Opción 2: Jupyter Notebook

```bash
jupyter notebook flujo_completo.ipynb
```

El notebook incluye:
- Generación de datos
- Visualizaciones
- Entrenamiento paso a paso
- Análisis de resultados

## 📦 Instalación

1. Crear un entorno virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Selección de Cantidad de Ejemplos

El proyecto soporta **3 tamaños de dataset**:

| Cantidad | Tiempo de entrenamiento | Uso recomendado |
|----------|------------------------|------------------|
| **100** | Rápido (segundos) | Pruebas y desarrollo |
| **500** | Medio (1-2 min) | **Recomendado** - Equilibrado |
| **1000** | Completo (5+ min) | Máxima precisión |

### En el script interactivo:

```python
python entrenar_mlp.py
# Te preguntará: ¿Qué cantidad deseas usar? (1/2/3) [default=2]
```

### En el notebook:

```python
# Celda de configuración (ajustar MODO_INTERACTIVO)
MODO_INTERACTIVO = True  # Input manual
# o
MODO_INTERACTIVO = False  # Usar cantidad predefinida
cantidad = 500  # Cambiar aquí: 100, 500 o 1000
```

## 🏗️ Estructura del Proyecto

```
perceptron2/
├── mlp.py                      # Clase MLP principal
├── generador_dataset.py        # Generación de datasets
├── distorsionador.py          # Distorsión inteligente (1s→0s)
├── clasificador.py            # Clasificador de letras
├── entrenar_mlp.py            # 🆕 Script interactivo de entrenamiento
├── comparar_distorsiones.py   # Comparación visual de métodos
├── flujo_completo.ipynb       # Notebook completo
├── requirements.txt           # Dependencias
└── data/
    ├── originales/
    │   ├── 100/letras.csv
    │   ├── 500/letras.csv
    │   └── 1000/letras.csv
    └── distorsionadas/
        ├── 100/letras.csv
        ├── 500/letras.csv
        └── 1000/letras.csv
```

## 🎯 Características Implementadas

### ✅ 1. Arquitecturas Flexibles

```python
from mlp import MLP

# 1 capa oculta (simple y rápida)
mlp = MLP(capas_ocultas=[8])  # 100 → 8 → 3

# 2 capas ocultas (más capacidad)
mlp = MLP(capas_ocultas=[10, 8])  # 100 → 10 → 8 → 3
```

**API Simplificada**: Solo especificas capas ocultas, entrada (100) y salida (3) son fijos.

### ✅ 2. Dos Métodos de Distorsión

#### Método Clásico:
```python
generador.generar_data_distorsionadas(cant=500, min_distorsion=0.01, max_distorsion=0.30)
```
- Inversión aleatoria (0↔1)

#### Método Distorsionador (Recomendado):
```python
generador.generar_data_distorsionadas_v2(cant=500, min_distorsion=5.0, max_distorsion=25.0)
```
- Intercambio inteligente (1s→0s)
- Más realista para degradación visual

### ✅ 3. Backpropagation Optimizado

- Operaciones vectorizadas (100-1000x más rápido)
- Separado en `backward_propagation()` y `gradiente_descendente()`
- Momentum estándar implementado correctamente

### ✅ 4. Monitoreo de Entrenamiento

```python
historial = mlp.entrenar(X, y, epochs=50, verbose=True)
# Muestra progreso en CADA época (no cada 100)
```

### ✅ 5. Selección Interactiva de Cantidad

**Tres opciones disponibles**:
- 🔹 **100 ejemplos**: Pruebas rápidas
- 🔹 **500 ejemplos**: Equilibrado (recomendado)
- 🔹 **1000 ejemplos**: Dataset completo

## 📋 Requisitos del Proyecto (Cumplidos)

### Datasets
- 3 datasets con 100, 500 y 1000 ejemplos
- Patrones en matriz 10x10 (letras b, d, f)
- 10% patrones sin distorsión
- 90% con distorsión del 1% al 30%

### Entrenamiento
- 3 conjuntos de validación por dataset (10%, 20%, 30%)
- 1 o 2 capas ocultas
- 5 a 10 neuronas por capa
- Funciones de activación: lineal y sigmoidal
- Learning rate: 0 a 1
- Momentum: 0 a 1

### Reconocimiento
- Patrón distorsionado del 0% al 30% (generado automática o manualmente)

## Próximos Pasos

1. ✅ Implementar clase MLP con arquitectura configurable
2. ⏳ Generar datasets de patrones (b, d, f)
3. ⏳ Implementar interfaz de usuario
4. ⏳ Sistema de evaluación y métricas (MSE, error de entrenamiento, validación)
5. ⏳ Generación de informes

## Autor

Trabajo Práctico - Inteligencia Artificial 2025
