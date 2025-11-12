"""
Utilidades para visualización de resultados del entrenamiento del MLP.
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
