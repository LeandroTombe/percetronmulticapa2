"""
Script de entrenamiento del MLP con selección interactiva de parámetros.
Permite elegir cantidad de ejemplos (100/500/1000), distorsión, epochs, etc.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from generador_dataset import GeneradorDataset
from mlp import MLP
from clasificador import ClasificadorLetras
from visualizador import (graficar_mse_entrenamiento_validacion, 
                          imprimir_resumen_entrenamiento,
                          analizar_overfitting)

def leer_dataset(cantidad, tipo='originales'):
    """Lee un dataset desde CSV"""
    file_path = os.path.join('data', tipo, str(cantidad), 'letras.csv')
    df = pd.read_csv(file_path, sep=';', header=None)
    X = df.iloc[:, :100].values
    y = df.iloc[:, 100:].values
    return X, y

def entrenar_con_configuracion():
    """Entrena el MLP con configuración interactiva"""
    
    print("="*70)
    print("🧠 ENTRENADOR DE MLP - Clasificador de Letras (B, D, F)")
    print("="*70)
    
    # ========================================
    # 1. SELECCIONAR CANTIDAD DE EJEMPLOS
    # ========================================
    print("\n📊 PASO 1: Seleccionar cantidad de ejemplos")
    print("-"*70)
    print("   1️⃣  100 ejemplos  (rápido, ideal para pruebas)")
    print("   2️⃣  500 ejemplos  (equilibrado, recomendado)")
    print("   3️⃣  1000 ejemplos (dataset completo, más tiempo)")
    
    opcion = input("\n¿Qué cantidad deseas usar? (1/2/3) [default=2]: ").strip()
    
    if opcion == "1":
        cantidad = 100
    elif opcion == "3":
        cantidad = 1000
    else:
        cantidad = 500
    
    print(f"\n✅ Seleccionado: {cantidad} ejemplos")
    
    # ========================================
    # 2. SELECCIONAR PORCENTAJE DE DISTORSIÓN
    # ========================================
    print("\n🎲 PASO 2: Seleccionar distorsión")
    print("-"*70)
    print("   Distorsión baja (5-10%):  Datos más limpios")
    print("   Distorsión media (15-20%): Equilibrado")
    print("   Distorsión alta (25-30%):  Más desafío")
    
    dist_input = input("\n¿Qué % de distorsión? (1-30) [default=10]: ").strip()
    
    try:
        distorsion = int(dist_input) if dist_input else 10
        distorsion = max(1, min(30, distorsion))  # Limitar entre 1-30
    except ValueError:
        distorsion = 10
    
    print(f"\n✅ Seleccionado: {distorsion}% de distorsión")
    
    # ========================================
    # 3. GENERAR/VERIFICAR DATASETS
    # ========================================
    print("\n📁 PASO 3: Verificar datasets")
    print("-"*70)
    
    generador = GeneradorDataset()
    
    # Verificar si existen los archivos
    path_orig = os.path.join('data', 'originales', str(cantidad), 'letras.csv')
    path_dist = os.path.join('data', 'distorsionadas', str(cantidad), 'letras.csv')
    
    if not os.path.exists(path_orig):
        print(f"⚠️  Dataset original no encontrado. Generando...")
        generador.generar_data_letras(cantidad)
    else:
        print(f"✅ Dataset original encontrado: {path_orig}")
    
    # Preguntar si regenerar distorsionado
    if os.path.exists(path_dist):
        regenerar = input(f"\n¿Regenerar dataset distorsionado con {distorsion}%? (s/n) [n]: ").strip().lower()
        if regenerar == 's':
            generador.generar_data_con_distorsiones_especificas(cantidad, distorsion, mezclar=False)
    else:
        print(f"⚠️  Dataset distorsionado no encontrado. Generando...")
        generador.generar_data_con_distorsiones_especificas(cantidad, distorsion, mezclar=False)
    
    # ========================================
    # 4. CARGAR DATOS
    # ========================================
    print("\n📥 PASO 4: Cargar datos")
    print("-"*70)
    
    X_dist, y_dist = leer_dataset(cantidad, 'distorsionadas')
    
    print(f"✅ Datos cargados: {len(X_dist)} ejemplos")
    print(f"   - Shape X: {X_dist.shape}")
    print(f"   - Shape y: {y_dist.shape}")
    
    # ========================================
    # 4B. SELECCIONAR SPLIT DE VALIDACIÓN
    # ========================================
    print("\n🔀 PASO 4B: Configurar validación")
    print("-"*70)
    print("   El dataset se dividirá en:")
    print("   - Conjunto de entrenamiento (para que el modelo aprenda)")
    print("   - Conjunto de validación (para evaluar sin haber visto)")
    print("\n   Opciones:")
    print("   1️⃣  10% validación / 90% entrenamiento")
    print("   2️⃣  20% validación / 80% entrenamiento (recomendado)")
    print("   3️⃣  30% validación / 70% entrenamiento")
    
    opcion_val = input("\n¿Qué split deseas? (1/2/3) [default=2]: ").strip()
    
    if opcion_val == "1":
        porcentaje_validacion = 0.10
    elif opcion_val == "3":
        porcentaje_validacion = 0.30
    else:
        porcentaje_validacion = 0.20
    
    print(f"\n✅ Split seleccionado:")
    print(f"   - Validación: {int(porcentaje_validacion*100)}%")
    print(f"   - Entrenamiento: {int((1-porcentaje_validacion)*100)}%")
    
    # Dividir datos usando train_test_split con estratificación
    X_train, X_val, y_train, y_val = train_test_split(
        X_dist, y_dist,
        test_size=porcentaje_validacion,
        random_state=42,  # Para reproducibilidad
        stratify=np.argmax(y_dist, axis=1)  # Mantener proporción B/D/F
    )
    
    # Mezclar solo el conjunto de entrenamiento
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"\n📊 Datos divididos:")
    print(f"   - Entrenamiento: {len(X_train)} ejemplos")
    print(f"   - Validación:    {len(X_val)} ejemplos")
    
    # ========================================
    # 5. CONFIGURAR ARQUITECTURA MLP
    # ========================================
    print("\n🏗️  PASO 5: Configurar arquitectura")
    print("-"*70)
    print("   Arquitectura base: 100 → [ocultas] → 3")
    print("\n   🆕 Usando la NUEVA API del MLP")
    print("   Opciones de configuración:")
    print("   1️⃣  1 capa  - 8 neuronas")
    print("   2️⃣  1 capa  - 10 neuronas (recomendado)")
    print("   3️⃣  2 capas - 8 y 6 neuronas")
    print("   4️⃣  2 capas - 10 y 8 neuronas")
    
    opcion_arq = input("\n¿Qué arquitectura deseas? (1/2/3/4) [default=2]: ").strip()
    
    # Configuraciones usando la NUEVA API
    if opcion_arq == "1":
        num_capas = 1
        neuronas1 = 8
        neuronas2 = None
    elif opcion_arq == "3":
        num_capas = 2
        neuronas1 = 8
        neuronas2 = 6
    elif opcion_arq == "4":
        num_capas = 2
        neuronas1 = 10
        neuronas2 = 8
    else:  # default "2"
        num_capas = 1
        neuronas1 = 10
        neuronas2 = None
    
    if num_capas == 1:
        print(f"\n✅ Arquitectura: 100 → {neuronas1} → 3 (1 capa oculta)")
    else:
        print(f"\n✅ Arquitectura: 100 → {neuronas1} → {neuronas2} → 3 (2 capas ocultas)")
    
    # ========================================
    # 6. CONFIGURAR HIPERPARÁMETROS
    # ========================================
    print("\n⚙️  PASO 6: Configurar hiperparámetros")
    print("-"*70)
    
    lr_input = input("Learning rate (0.01-0.5) [default=0.1]: ").strip()
    try:
        learning_rate = float(lr_input) if lr_input else 0.1
        learning_rate = max(0.01, min(0.5, learning_rate))
    except ValueError:
        learning_rate = 0.1
    
    momentum_input = input("Momentum (0.0-0.99) [default=0.9]: ").strip()
    try:
        momentum = float(momentum_input) if momentum_input else 0.9
        momentum = max(0.0, min(0.99, momentum))
    except ValueError:
        momentum = 0.9
    
    epochs_input = input("Epochs (10-1000) [default=50]: ").strip()
    try:
        epochs = int(epochs_input) if epochs_input else 50
        epochs = max(10, min(1000, epochs))
    except ValueError:
        epochs = 50
    
    print(f"\n✅ Hiperparámetros:")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Momentum: {momentum}")
    print(f"   - Epochs: {epochs}")
    
    # ========================================
    # 7. CREAR Y ENTRENAR MLP CON VALIDACIÓN
    # ========================================
    print("\n🎯 PASO 7: Entrenar MLP con validación")
    print("-"*70)
    
    # Crear MLP usando la NUEVA API
    if num_capas == 1:
        mlp = MLP(
            capas_ocultas=1,
            cantidad_neuronas=neuronas1,
            learning_rate=learning_rate,
            momentum=momentum
        )
    else:  # 2 capas
        mlp = MLP(
            capas_ocultas=2,
            cantidad_neuronas1=neuronas1,
            cantidad_neuronas2=neuronas2,
            learning_rate=learning_rate,
            momentum=momentum
        )
    
    print(f"\n🏗️  Arquitectura creada: {mlp.arquitectura}")
    print(f"   - Entrada: 100 neuronas (matriz 10x10)")
    print(f"   - Capas ocultas: {mlp.capas_ocultas}")
    print(f"   - Salida: 3 neuronas (B, D, F)")
    
    print(f"\n🏃 Iniciando entrenamiento con validación...")
    print(f"   (Esto mostrará el error de entrenamiento Y validación época por época)")
    print()
    
    # Entrenar CON datos de validación
    historial = mlp.entrenar(
        X_train, y_train,              # Datos de entrenamiento
        X_val=X_val, y_val=y_val,      # ← NUEVO: datos de validación
        epochs=epochs, 
        verbose=True
    )
    
    # Imprimir resumen automático
    imprimir_resumen_entrenamiento(historial)
    
    # Detectar overfitting
    analisis = analizar_overfitting(historial)
    if "OVERFITTING" in analisis['estado']:
        print(f"⚠️  RECOMENDACIÓN:")
        print(f"   - Reducir épocas de entrenamiento")
        print(f"   - Reducir learning_rate")
        print(f"   - Aumentar el dataset")
    
    # ========================================
    # 7B. GRAFICAR CURVAS MSE
    # ========================================
    print("\n📊 Generando gráfica MSE...")
    graficar_mse_entrenamiento_validacion(
        historial,
        titulo=f"MSE - {mlp.arquitectura} - LR:{learning_rate} - Mom:{momentum}"
    )
    
    # ========================================
    # 8. EVALUAR MODELO EN TRAIN Y VALIDACIÓN
    # ========================================
    print("\n📊 PASO 8: Evaluar modelo")
    print("-"*70)
    
    clasificador = ClasificadorLetras(mlp)
    letras_map = {0: 'B', 1: 'D', 2: 'F'}
    
    # Evaluar en ENTRENAMIENTO
    print("\n🔹 Evaluación en conjunto de ENTRENAMIENTO:")
    aciertos_train = 0
    for i in range(len(X_train)):
        resultado = clasificador.clasificar_patron(X_train[i])
        real = letras_map[np.argmax(y_train[i])]
        if resultado['letra'] == real:
            aciertos_train += 1
    
    precision_train = (aciertos_train / len(X_train)) * 100
    print(f"   Precisión: {precision_train:.2f}% ({aciertos_train}/{len(X_train)})")
    
    # Evaluar en VALIDACIÓN
    print("\n🔹 Evaluación en conjunto de VALIDACIÓN:")
    aciertos_val = 0
    for i in range(len(X_val)):
        resultado = clasificador.clasificar_patron(X_val[i])
        real = letras_map[np.argmax(y_val[i])]
        if resultado['letra'] == real:
            aciertos_val += 1
    
    precision_val = (aciertos_val / len(X_val)) * 100
    print(f"   Precisión: {precision_val:.2f}% ({aciertos_val}/{len(X_val)})")
    
    # Análisis
    print(f"\n📈 ANÁLISIS:")
    print(f"   Diferencia: {abs(precision_train - precision_val):.2f}%")
    if precision_train - precision_val > 10:
        print(f"   ⚠️  Posible overfitting (modelo memoriza en lugar de aprender)")
    elif precision_val > precision_train:
        print(f"   ✅ Excelente! El modelo generaliza bien")
    else:
        print(f"   ✅ Buen balance entre entrenamiento y validación")
    
    # ========================================
    # 9. GUARDAR MODELO (OPCIONAL)
    # ========================================
    print("\n💾 PASO 9: Guardar modelo")
    print("-"*70)
    
    guardar = input("¿Deseas guardar el modelo? (s/n) [s]: ").strip().lower()
    
    if guardar != 'n':
        nombre = input("Nombre del archivo [modelo_mlp.json]: ").strip()
        if not nombre:
            nombre = "modelo_mlp.json"
        if not nombre.endswith('.json'):
            nombre += '.json'
        
        mlp.guardar_modelo(nombre)
        print(f"✅ Modelo guardado: {nombre}")
    
    print("\n" + "="*70)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    return mlp, historial, clasificador

if __name__ == "__main__":
    try:
        mlp, historial, clasificador = entrenar_con_configuracion()
    except KeyboardInterrupt:
        print("\n\n❌ Entrenamiento cancelado por el usuario.")
    except Exception as e:
        print(f"\n\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
