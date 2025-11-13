"""
Utilidades para visualización de resultados del entrenamiento del MLP.

Responsabilidades:
- Visualizar curvas de aprendizaje (MSE)
- Visualizar distribuciones de datos
- Visualizar patrones de letras
- Visualizar métricas de rendimiento
- Crear matrices de confusión
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union


def graficar_mse_entrenamiento_validacion(historial: Dict[str, List[float]], 
                                          titulo: str = "MSE de Entrenamiento vs Validación",
                                          guardar_como: str = None):
    """
    Grafica las curvas de MSE para entrenamiento y validación.
    
    Parámetros:
    -----------
    historial : dict
        Diccionario con keys 'train_loss' y 'val_loss'
    titulo : str
        Título del gráfico
    guardar_como : str, opcional
        Ruta del archivo para guardar la imagen
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(historial['train_loss']) + 1)
    
    plt.plot(epochs, historial['train_loss'], 'g-', label='Entrenamiento', linewidth=2)
    plt.plot(epochs, historial['val_loss'], 'b-', label='Validación', linewidth=2)
    
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Error global', fontsize=12)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Agregar anotaciones de valores inicial y final
    plt.annotate(f'{historial["train_loss"][0]:.4f}', 
                xy=(1, historial['train_loss'][0]), 
                xytext=(5, 10), textcoords='offset points', fontsize=9)
    plt.annotate(f'{historial["train_loss"][-1]:.4f}', 
                xy=(len(epochs), historial['train_loss'][-1]), 
                xytext=(5, 10), textcoords='offset points', fontsize=9)
    
    plt.annotate(f'{historial["val_loss"][0]:.4f}', 
                xy=(1, historial['val_loss'][0]), 
                xytext=(5, -15), textcoords='offset points', fontsize=9)
    plt.annotate(f'{historial["val_loss"][-1]:.4f}', 
                xy=(len(epochs), historial['val_loss'][-1]), 
                xytext=(5, -15), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfica guardada en: {guardar_como}")
    
    plt.show()


def graficar_comparacion_splits(resultados: Dict[str, Dict], 
                                titulo: str = "Comparación de Splits de Validación"):
    """
    Grafica la comparación de diferentes splits de validación.
    
    Parámetros:
    -----------
    resultados : dict
        Diccionario con formato:
        {
            '10%': {'train_loss': [...], 'val_loss': [...]},
            '20%': {'train_loss': [...], 'val_loss': [...]},
            '30%': {'train_loss': [...], 'val_loss': [...]}
        }
    titulo : str
        Título del gráfico
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    
    colores = {'train': 'g', 'val': 'b'}
    
    for idx, (split_name, historial) in enumerate(resultados.items()):
        ax = axes[idx]
        epochs = range(1, len(historial['train_loss']) + 1)
        
        ax.plot(epochs, historial['train_loss'], f'{colores["train"]}-', 
                label='Entrenamiento', linewidth=2)
        ax.plot(epochs, historial['val_loss'], f'{colores["val"]}-', 
                label='Validación', linewidth=2)
        
        ax.set_xlabel('Épocas', fontsize=11)
        ax.set_ylabel('Error global', fontsize=11)
        ax.set_title(f'Split: {split_name} validación', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Calcular gap final
        gap = historial['val_loss'][-1] - historial['train_loss'][-1]
        ax.text(0.5, 0.95, f'Gap final: {gap:.4f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def graficar_mse_simple(historial: Union[List[float], Dict[str, List[float]]], 
                       titulo: str = "Evolución del Error (MSE)",
                       guardar_como: str = None):
    """
    Grafica la evolución del MSE (solo entrenamiento o con validación).
    
    Parámetros:
    -----------
    historial : list o dict
        Lista de errores [e1, e2, ...] o dict {'train_loss': [...], 'val_loss': [...]}
    titulo : str
        Título del gráfico
    guardar_como : str, opcional
        Ruta del archivo para guardar la imagen
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(historial, dict):
        # Historial con validación
        epochs = range(1, len(historial['train_loss']) + 1)
        plt.plot(epochs, historial['train_loss'], 'g-', label='Entrenamiento', linewidth=2)
        plt.plot(epochs, historial['val_loss'], 'b-', label='Validación', linewidth=2)
        plt.legend(loc='upper right', fontsize=11)
    else:
        # Historial solo de entrenamiento
        epochs = range(1, len(historial) + 1)
        plt.plot(epochs, historial, 'g-', label='Error', linewidth=2)
    
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Error (MSE)', fontsize=12)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if guardar_como:
        plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfica guardada en: {guardar_como}")
    
    plt.show()


def analizar_overfitting(historial: Dict[str, List[float]], 
                        umbral_gap: float = 0.05) -> Dict:
    """
    Analiza si hay overfitting comparando train y validation loss.
    
    Parámetros:
    -----------
    historial : dict
        Diccionario con 'train_loss' y 'val_loss'
    umbral_gap : float
        Umbral para considerar overfitting
    
    Retorna:
    --------
    dict
        Diccionario con análisis del overfitting
    """
    train_final = historial['train_loss'][-1]
    val_final = historial['val_loss'][-1]
    gap = val_final - train_final
    
    # Calcular mejora
    train_inicial = historial['train_loss'][0]
    val_inicial = historial['val_loss'][0]
    mejora_train = ((train_inicial - train_final) / train_inicial) * 100
    mejora_val = ((val_inicial - val_final) / val_inicial) * 100
    
    # Determinar estado
    if gap > umbral_gap:
        estado = "⚠️  OVERFITTING DETECTADO"
        descripcion = "El modelo se ajusta demasiado a los datos de entrenamiento"
    elif val_final < train_final:
        estado = "✅ EXCELENTE"
        descripcion = "El modelo generaliza mejor de lo esperado"
    else:
        estado = "✅ BUEN BALANCE"
        descripcion = "Balance saludable entre entrenamiento y validación"
    
    return {
        'estado': estado,
        'descripcion': descripcion,
        'gap': gap,
        'train_final': train_final,
        'val_final': val_final,
        'mejora_train': mejora_train,
        'mejora_val': mejora_val
    }


def imprimir_resumen_entrenamiento(historial: Dict[str, List[float]]):
    """
    Imprime un resumen detallado del entrenamiento.
    
    Parámetros:
    -----------
    historial : dict
        Diccionario con 'train_loss' y 'val_loss'
    """
    analisis = analizar_overfitting(historial)
    
    print("\n" + "="*70)
    print("📊 RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    
    print(f"\n📉 Error Entrenamiento:")
    print(f"   - Inicial: {historial['train_loss'][0]:.6f}")
    print(f"   - Final:   {historial['train_loss'][-1]:.6f}")
    print(f"   - Mejora:  {analisis['mejora_train']:.2f}%")
    
    print(f"\n📉 Error Validación:")
    print(f"   - Inicial: {historial['val_loss'][0]:.6f}")
    print(f"   - Final:   {historial['val_loss'][-1]:.6f}")
    print(f"   - Mejora:  {analisis['mejora_val']:.2f}%")
    
    print(f"\n🎯 Análisis:")
    print(f"   - Gap (val - train): {analisis['gap']:.6f}")
    print(f"   - Estado: {analisis['estado']}")
    print(f"   - {analisis['descripcion']}")
    
    print("="*70 + "\n")


def visualizar_ejemplos_dataset(generador, X, y, letras_map, num_ejemplos=6):
    """
    Visualiza ejemplos del dataset comparando originales vs distorsionados.
    
    Parámetros:
    -----------
    generador : GeneradorDataset
        Instancia del generador para obtener patrones originales
    X : np.ndarray
        Patrones distorsionados
    y : np.ndarray
        Etiquetas one-hot
    letras_map : dict
        Mapeo de índices a letras {0: 'B', 1: 'D', 2: 'F'}
    num_ejemplos : int
        Número de ejemplos a mostrar
    """
    fig, axes = plt.subplots(2, num_ejemplos, figsize=(15, 5))
    
    for i in range(num_ejemplos):
        # Obtener la letra real del ejemplo distorsionado
        letra_dist = letras_map[np.argmax(y[i])]
        
        # Generar la versión PERFECTA de esa misma letra
        X_orig_letra = generador.generar_letra(letra_dist)
        
        # Fila superior: letra perfecta
        axes[0, i].imshow(X_orig_letra.reshape(10, 10), cmap='binary')
        axes[0, i].set_title(f'Original: {letra_dist}')
        axes[0, i].axis('off')
        
        # Fila inferior: letra distorsionada
        axes[1, i].imshow(X[i].reshape(10, 10), cmap='binary')
        # Calcular distorsión comparando con la versión perfecta
        dist = np.sum(X[i] != X_orig_letra) / 100 * 100
        axes[1, i].set_title(f'Dist: {dist:.1f}%')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def graficar_comparacion_desempeno(acc_train, acc_val, X_train, X_val, y_pred_train, y_pred_val, y_train, y_val):
    """
    Grafica comparación de desempeño entre entrenamiento y validación.
    
    Parámetros:
    -----------
    acc_train, acc_val : float
        Exactitudes de entrenamiento y validación
    X_train, X_val : np.ndarray
        Datos de entrenamiento y validación
    y_pred_train, y_pred_val : np.ndarray
        Predicciones del modelo
    y_train, y_val : np.ndarray
        Etiquetas verdaderas
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Barras de exactitud
    datasets = ['Entrenamiento', 'Validación']
    accuracies = [acc_train * 100, acc_val * 100]
    colors = ['#3498db', '#e74c3c']
    
    bars = axes[0].bar(datasets, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Exactitud (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Exactitud por Conjunto', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Agregar valores sobre las barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Agregar línea de referencia en 100%
    axes[0].axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100%')
    axes[0].legend()
    
    # Gráfica 2: Predicciones correctas vs incorrectas
    correctas_train = int(acc_train * len(X_train))
    incorrectas_train = len(X_train) - correctas_train
    correctas_val = int(acc_val * len(X_val))
    incorrectas_val = len(X_val) - correctas_val
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, [correctas_train, correctas_val], width, 
                        label='Correctas', color='#2ecc71', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, [incorrectas_train, incorrectas_val], width,
                        label='Incorrectas', color='#e74c3c', alpha=0.8)
    
    axes[1].set_ylabel('Cantidad de Predicciones', fontsize=12, fontweight='bold')
    axes[1].set_title('Predicciones Correctas vs Incorrectas', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Entrenamiento', 'Validación'])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Agregar valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Resumen
    print(f"\n📈 Resumen de Desempeño:")
    print(f"   Entrenamiento: {correctas_train}/{len(X_train)} correctas")
    print(f"   Validación: {correctas_val}/{len(X_val)} correctas")
    diferencia = abs(acc_train - acc_val)*100
    estado = '✅ Buena generalización' if diferencia < 15 else '⚠️ Posible overfitting'
    print(f"   Diferencia: {diferencia:.2f}% {estado}")


def visualizar_clasificacion_letra(clasificador, letra, distorsion=20):
    """
    Visualiza un ejemplo de clasificación con distorsión.
    
    Parámetros:
    -----------
    clasificador : ClasificadorLetras
        Instancia del clasificador entrenado
    letra : str
        Letra a clasificar ('B', 'D', o 'F')
    distorsion : int
        Porcentaje de distorsión (0-30)
    """
    # Generar patrones
    patron_original = clasificador.generador.generar_letra(letra).reshape(10, 10)
    patron_distorsionado = clasificador.generar_patron_distorsionado(letra, distorsion)
    patron_dist_2d = patron_distorsionado.reshape(10, 10)
    
    # Clasificar
    resultado = clasificador.clasificar_patron(patron_distorsionado)
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Patrón original
    axes[0].imshow(patron_original, cmap='binary', interpolation='nearest')
    axes[0].set_title(f'Patrón Original: {letra}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].grid(True, alpha=0.3)
    
    # Patrón distorsionado
    axes[1].imshow(patron_dist_2d, cmap='binary', interpolation='nearest')
    axes[1].set_title(f'Distorsión: {distorsion}%', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].grid(True, alpha=0.3)
    
    # Resultado de clasificación
    axes[2].axis('off')
    resultado_texto = f"""
    RESULTADO DE CLASIFICACIÓN
    
    Letra Real:       {letra}
    Predicción:       {resultado['letra']}
    Estado:           {'✅ CORRECTO' if resultado['letra'] == letra else '❌ INCORRECTO'}
    
    Confianza:        {resultado['confianza']*100:.1f}%
    
    PROBABILIDADES:
    ─────────────────
    B: {resultado['probabilidades'][0]*100:.1f}%
    D: {resultado['probabilidades'][1]*100:.1f}%
    F: {resultado['probabilidades'][2]*100:.1f}%
    """
    
    axes[2].text(0.1, 0.5, resultado_texto, fontsize=12, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def visualizar_ejemplos_originales_vs_distorsionados(generador, X_dist, y_dist, letras_map, num_ejemplos=6):
    """
    Visualiza ejemplos de letras originales vs distorsionadas.
    
    Parámetros:
    -----------
    generador : GeneradorDataset
        Instancia del generador de datasets
    X_dist : numpy.ndarray
        Array de patrones distorsionados
    y_dist : numpy.ndarray
        Array de etiquetas one-hot
    letras_map : dict
        Diccionario {índice: letra}
    num_ejemplos : int
        Número de ejemplos a visualizar
    """
    fig, axes = plt.subplots(2, num_ejemplos, figsize=(15, 5))
    
    for i in range(num_ejemplos):
        # Obtener la letra real del ejemplo distorsionado
        letra_dist = letras_map[np.argmax(y_dist[i])]
        
        # Generar la versión PERFECTA de esa misma letra
        X_orig_letra = generador.generar_letra(letra_dist)
        
        # Fila superior: letra perfecta
        axes[0, i].imshow(X_orig_letra.reshape(10, 10), cmap='binary')
        axes[0, i].set_title(f'Original: {letra_dist}', fontsize=10, fontweight='bold')
        axes[0, i].axis('off')
        
        # Fila inferior: letra distorsionada
        axes[1, i].imshow(X_dist[i].reshape(10, 10), cmap='binary')
        
        # CORRECCIÓN: Calcular distorsión sobre 100 píxeles TOTALES (nueva lógica)
        pixeles_cambiados = np.sum(X_dist[i] != X_orig_letra)
        
        if pixeles_cambiados == 0:
            # Es perfecto
            axes[1, i].set_title(f'OK: 0% (Perfecto)', fontsize=9, fontweight='bold', color='green')
        else:
            # Calcular porcentaje sobre 100 píxeles totales
            pct_distorsion = (pixeles_cambiados / 100.0) * 100
            
            # Color según nivel de distorsión
            if pct_distorsion < 10:
                color = 'green'
            elif pct_distorsion < 20:
                color = 'orange'
            else:
                color = 'red'
            
            axes[1, i].set_title(f'Dist: {pct_distorsion:.1f}% ({pixeles_cambiados}px)', 
                               fontsize=9, fontweight='bold', color=color)
        
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluar_y_comparar(clasificador, X_train, y_train, X_val, y_val, letras_map):
    """
    Evalúa el clasificador en entrenamiento y validación, mostrando comparación.
    
    Parámetros:
    -----------
    clasificador : ClasificadorLetras
        Clasificador entrenado
    X_train, y_train : numpy.ndarray
        Datos de entrenamiento
    X_val, y_val : numpy.ndarray
        Datos de validación
    letras_map : dict
        Diccionario {índice: letra}
        
    Returns:
    --------
    dict : Diccionario con métricas {'precision_train', 'precision_val', 'aciertos_train', 'aciertos_val'}
    """
    # Evaluar en ENTRENAMIENTO
    print("🔹 Evaluación en conjunto de ENTRENAMIENTO:")
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
    
    return {
        'precision_train': precision_train,
        'precision_val': precision_val,
        'aciertos_train': aciertos_train,
        'aciertos_val': aciertos_val
    }


def visualizar_predicciones_aleatorias(clasificador, X_dist, y_dist, letras_map, num_ejemplos=12):
    """
    Visualiza predicciones aleatorias del clasificador.
    
    Parámetros:
    -----------
    clasificador : ClasificadorLetras
        Clasificador entrenado
    X_dist : numpy.ndarray
        Array de patrones
    y_dist : numpy.ndarray
        Array de etiquetas one-hot
    letras_map : dict
        Diccionario {índice: letra}
    num_ejemplos : int
        Número de ejemplos a mostrar
    """
    indices_random = np.random.choice(len(X_dist), num_ejemplos, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, i in enumerate(indices_random):
        resultado = clasificador.clasificar_patron(X_dist[i])
        pred = resultado['letra']
        real = letras_map[np.argmax(y_dist[i])]
        
        # Calcular distorsión sobre 100 píxeles TOTALES (nueva lógica)
        X_orig_letra = clasificador.generador.generar_letra(real)
        pixeles_cambiados = np.sum(X_dist[i] != X_orig_letra)
        
        if pixeles_cambiados == 0:
            distorsion_text = "[OK] 0%"
            pct_distorsion = 0.0
        else:
            pct_distorsion = (pixeles_cambiados / 100.0) * 100
            distorsion_text = f"Dist: {pct_distorsion:.1f}%"
        
        correcto = pred == real
        estado = "[OK]" if correcto else "[X]"
        color = 'green' if correcto else 'red'
        
        axes[idx].imshow(X_dist[i].reshape(10, 10), cmap='binary')
        
        # Título con estado, real, predicción y distorsión
        titulo = f'{estado} Real:{real} | Pred:{pred}\n{distorsion_text} | Conf:{resultado["confianza"]*100:.0f}%'
        axes[idx].set_title(titulo, color=color, fontweight='bold', fontsize=9)
        axes[idx].axis('off')
    
    plt.suptitle('Predicciones (Verde=OK, Rojo=Error)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return resultado


def listar_dataset_completo(generador, X_dist, y_dist, letras_map):
    """
    Muestra un listado completo de todos los ejemplos del dataset con su letra y distorsión.
    
    Parámetros:
    -----------
    generador : GeneradorDataset
        Instancia del generador con los patrones originales
    X_dist : numpy.ndarray
        Array de patrones del dataset
    y_dist : numpy.ndarray
        Array de etiquetas one-hot
    letras_map : dict
        Diccionario {índice: letra}
    """
    import pandas as pd
    
    # Obtener patrones originales
    patron_B = generador.generar_letra('B')
    patron_D = generador.generar_letra('D')
    patron_F = generador.generar_letra('F')
    
    patrones_originales = {
        'B': patron_B,
        'D': patron_D,
        'F': patron_F
    }
    
    # Crear lista de datos
    datos = []
    for i in range(len(X_dist)):
        letra_real = letras_map[np.argmax(y_dist[i])]
        patron_original = patrones_originales[letra_real]
        patron_actual = X_dist[i]
        
        # Calcular porcentaje de distorsión sobre 100 píxeles TOTALES
        # Nueva lógica: distorsión = (cambios / 100) * 100
        diferencias = np.sum(patron_original != patron_actual)
        
        # IMPORTANTE: Ahora calculamos sobre 100 píxeles totales, no sobre píxeles activos
        # Ejemplo: 30 cambios de píxeles = 30% de distorsión
        porcentaje_distorsion = (diferencias / 100.0) * 100
        
        # Determinar si es perfecto o distorsionado
        tipo = "Perfecto" if diferencias == 0 else "Distorsionado"
        
        datos.append({
            '#': i + 1,
            'Letra': letra_real,
            'Tipo': tipo,
            'Distorsión': f"{porcentaje_distorsion:.1f}%",
            'Píxeles cambiados': diferencias
        })
    
    # Crear DataFrame
    df = pd.DataFrame(datos)
    
    # Contar por letra y tipo
    perfectos = df[df['Tipo'].str.contains('Perfecto')]
    distorsionados = df[df['Tipo'].str.contains('Distorsionado')]
    
    # Crear visualización colorida
    print("="*80)
    print("📊 LISTADO COMPLETO DEL DATASET")
    print("="*80)
    print()
    
    # Resumen por letra
    print("📈 RESUMEN POR LETRA:")
    print("-" * 80)
    for letra in ['B', 'D', 'F']:
        letra_df = df[df['Letra'] == letra]
        perf = len(letra_df[letra_df['Tipo'].str.contains('Perfecto')])
        dist = len(letra_df[letra_df['Tipo'].str.contains('Distorsionado')])
        print(f"   {letra}: {len(letra_df):3d} ejemplos | ✨ {perf:2d} perfectos | 🔀 {dist:3d} distorsionados")
    
    print()
    print(f"   TOTAL: {len(df)} ejemplos | ✨ {len(perfectos)} perfectos | 🔀 {len(distorsionados)} distorsionados")
    print()
    
    # Estadísticas de distorsión
    print("📊 ESTADÍSTICAS DE DISTORSIÓN:")
    print("-" * 80)
    if len(distorsionados) > 0:
        dist_valores = distorsionados['Distorsión'].str.rstrip('%').astype(float)
        print(f"   Mínima: {dist_valores.min():.1f}%")
        print(f"   Máxima: {dist_valores.max():.1f}%")
        print(f"   Promedio: {dist_valores.mean():.1f}%")
        print(f"   Mediana: {dist_valores.median():.1f}%")
    print()
    
    # Mostrar tabla completa con paginación
    print("📋 DETALLE DE TODOS LOS EJEMPLOS:")
    print("="*80)
    
    # Agrupar por letra para mejor visualización
    for letra in ['B', 'D', 'F']:
        letra_df = df[df['Letra'] == letra]
        print(f"\n{'='*80}")
        print(f"   LETRA {letra} ({len(letra_df)} ejemplos)")
        print(f"{'='*80}")
        
        # Mostrar perfectos primero
        perf_letra = letra_df[letra_df['Tipo'].str.contains('Perfecto')]
        if len(perf_letra) > 0:
            print(f"\n   ✨ PERFECTOS ({len(perf_letra)}):")
            print(f"   {'-'*76}")
            for idx, row in perf_letra.iterrows():
                print(f"   #{row['#']:3d} | {row['Letra']} | {row['Tipo']:15s} | {row['Distorsión']:>6s}")
        
        # Luego mostrar distorsionados
        dist_letra = letra_df[letra_df['Tipo'].str.contains('Distorsionado')]
        if len(dist_letra) > 0:
            print(f"\n   🔀 DISTORSIONADOS ({len(dist_letra)}):")
            print(f"   {'-'*76}")
            for idx, row in dist_letra.iterrows():
                # Color según nivel de distorsión
                dist_val = float(row['Distorsión'].rstrip('%'))
                if dist_val < 10:
                    emoji = "🟢"  # Baja
                elif dist_val < 20:
                    emoji = "🟡"  # Media
                else:
                    emoji = "🔴"  # Alta
                
                print(f"   #{row['#']:3d} | {row['Letra']} | {emoji} {row['Distorsión']:>6s} | {row['Píxeles cambiados']:2d} píxeles cambiados")
    
    print()
    print("="*80)
    print("✅ Listado completo generado")
    print("="*80)
    
    return df
