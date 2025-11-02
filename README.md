# Perceptrón Multicapa (MLP) - TP 2025

Implementación de un Perceptrón Multicapa para clasificación de patrones según los requisitos del trabajo práctico.

## Instalación

1. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
venv\Scripts\activate
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

- `mlp.py`: Implementación principal del MLP
- `requirements.txt`: Dependencias del proyecto
- `datasets.py`: (próximamente) Generación de datasets
- `main.py`: (próximamente) Interfaz de usuario y experimentos

## Características Implementadas

### ✅ 1. Definición de Arquitectura

El MLP permite definir completamente su arquitectura:

- **Capas**: 1 o 2 capas ocultas (según requisitos)
- **Neuronas**: 5 a 10 neuronas por capa oculta
- **Funciones de activación**: lineal y sigmoidal

### Ejemplo de uso:

```python
from mlp import MLP

# Red con 1 capa oculta
mlp = MLP(
    arquitectura=[100, 10, 10],  # [entrada, oculta, salida]
    funciones_activacion=['sigmoidal', 'lineal'],
    learning_rate=0.1,
    momentum=0.5
)

# Red con 2 capas ocultas
mlp = MLP(
    arquitectura=[100, 8, 6, 10],  # [entrada, oculta1, oculta2, salida]
    funciones_activacion=['sigmoidal', 'sigmoidal', 'lineal'],
    learning_rate=0.1,
    momentum=0.9
)
```

## Requisitos del Proyecto

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
