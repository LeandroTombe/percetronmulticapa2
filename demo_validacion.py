"""
Script de prueba rápida del sistema de validación.
Genera datos, entrena con validación y muestra gráficas.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from generador_dataset import GeneradorDataset
from mlp import MLP
from visualizador import (graficar_mse_entrenamiento_validacion, 
                          imprimir_resumen_entrenamiento)

print("="*70)
print("🧪 DEMO: Sistema de Validación del MLP")
print("="*70)

# 1. Generar dataset pequeño para demo rápida
print("\n📦 Paso 1: Generando dataset de 200 ejemplos...")
generador = GeneradorDataset()
generador.generar_dataset(cantidad=200, porcentaje_distorsion=30, carpeta='demo')

# 2. Cargar datos
print("\n📥 Paso 2: Cargando datos...")
df = generador.leer_dataset_csv('demo/200/letras.csv')
X = df.iloc[:, :-1].values
y_labels = df.iloc[:, -1].values

# Convertir a one-hot
y = np.eye(3)[y_labels]

print(f"✅ Datos cargados: {len(X)} ejemplos")

# 3. Dividir en train/validación (20%)
print("\n🔀 Paso 3: Dividiendo en Train/Validación (20%)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y_labels
)

print(f"   - Entrenamiento: {len(X_train)} ejemplos (80%)")
print(f"   - Validación:    {len(X_val)} ejemplos (20%)")

# 4. Crear MLP con nueva API
print("\n🏗️  Paso 4: Creando MLP...")
mlp = MLP(
    capas_ocultas=1,
    cantidad_neuronas=8,
    learning_rate=0.4,
    momentum=0.6
)

print(f"   Arquitectura: {mlp.arquitectura}")

# 5. Entrenar CON validación
print("\n🏃 Paso 5: Entrenando con validación (30 épocas)...")
print("-"*70)

historial = mlp.entrenar(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=30,
    verbose=True
)

# 6. Mostrar resumen
print("\n" + "-"*70)
imprimir_resumen_entrenamiento(historial)

# 7. Graficar MSE
print("📊 Paso 6: Generando gráfica MSE...")
graficar_mse_entrenamiento_validacion(
    historial,
    titulo=f"Demo: MSE - {mlp.arquitectura} - LR:{mlp.learning_rate} - Mom:{mlp.momentum}"
)

print("\n✅ Demo completada exitosamente!")
print("="*70)
