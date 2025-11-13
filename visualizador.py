"""
Utilidades para visualización de resultados del entrenamiento del MLP.
"""

import matplotlib.pyplot as plt
from typing import Dict


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

