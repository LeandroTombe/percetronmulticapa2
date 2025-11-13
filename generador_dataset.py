"""
Generador de datasets de letras (B, D, F) con distorsión para entrenar MLP.
"""

import numpy as np
import pandas as pd
import os
import math
from random import shuffle


class GeneradorDataset:
    """
    Genera datasets de letras (B, D, F) con distorsión controlada.
    
    Responsabilidades:
    - Generar patrones de letras B, D, F
    - Aplicar distorsión aleatoria a los patrones
    - Generar datasets equilibrados
    - Guardar y cargar datasets en formato CSV
    """
    
    def __init__(self):
        self.letras = ['B', 'D', 'F']
        self.c_letras = {
            'B': np.array([1, 0, 0]),
            'D': np.array([0, 1, 0]),
            'F': np.array([0, 0, 1])
        }
        self.base_path = os.path.abspath('')
        self._crear_directorios()
    
    def _crear_directorios(self):
        """Crea las carpetas necesarias para guardar los datasets distorsionados"""
        # Solo crear carpetas para los tamaños que el usuario puede seleccionar
        for cant in [100, 500, 1000]:
            path = os.path.join(self.base_path, "data", "distorsionadas", str(cant))
            os.makedirs(path, exist_ok=True)
    
    def generar_letra(self, letra):
        """
        Genera una letra sin distorsión (patrón base).
        
        Args:
            letra: 'B', 'D' o 'F'
            
        Returns:
            Array 1D de 100 elementos (patrón 10x10 aplanado)
        """
        if letra == "B":
            letra_codigo = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        elif letra == "D":
            letra_codigo = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        else:  # F
            letra_codigo = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        
        return letra_codigo.flatten()
    
    def aplicar_distorsion(self, patron, porcentaje_distorsion, modo='mixto'):
        """
        Aplica distorsión sobre el total de 100 píxeles (modo mixto siempre).
        
        Args:
            patron: Array 1D de 100 elementos
            porcentaje_distorsion: Porcentaje de los 100 píxeles totales a distorsionar (0.01-0.30)
                                   Ejemplo: 0.30 = 30 cambios, 0.20 = 20 cambios, 0.10 = 10 cambios
            modo: (Ignorado, siempre usa modo mixto)
            
        Returns:
            Patrón distorsionado (copia, no modifica el original)
        
        Lógica mixta:
        - Combina aleatoriamente: apagar (1→0), prender (0→1) e intercambiar píxeles
        - El número de cambios se calcula sobre los 100 píxeles totales
        - Ejemplo: 30% de distorsión = 30 cambios de píxeles
        """
        patron_distorsionado = patron.copy()
        
        # 1. Calcular número de cambios sobre 100 píxeles totales
        num_cambios = int(100 * porcentaje_distorsion)
        
        # Garantizar al menos 1 cambio si se pidió distorsión
        if porcentaje_distorsion > 0 and num_cambios == 0:
            num_cambios = 1
        
        if num_cambios == 0:
            return patron_distorsionado
        
        # 2. Identificar píxeles disponibles
        indices_activos = np.where(patron == 1)[0]  # Píxeles que se pueden apagar
        indices_apagados = np.where(patron == 0)[0]  # Píxeles que se pueden prender
        indices_totales = np.arange(100)  # Todos los píxeles para intercambiar
        
        # 3. Dividir cambios en tres operaciones: apagar, prender, intercambiar
        # Distribución aproximadamente igual entre las 3 operaciones
        num_apagar = num_cambios // 3
        num_prender = num_cambios // 3
        num_intercambiar = num_cambios - num_apagar - num_prender  # El resto
        
        cambios_realizados = 0
        indices_ya_modificados = set()
        
        # 3a. Apagar píxeles (1 → 0)
        if num_apagar > 0 and len(indices_activos) > 0:
            num_apagar_real = min(num_apagar, len(indices_activos))
            indices_para_apagar = np.random.choice(indices_activos, num_apagar_real, replace=False)
            patron_distorsionado[indices_para_apagar] = 0
            indices_ya_modificados.update(indices_para_apagar)
            cambios_realizados += num_apagar_real
        
        # 3b. Prender píxeles (0 → 1)
        if num_prender > 0 and len(indices_apagados) > 0:
            num_prender_real = min(num_prender, len(indices_apagados))
            indices_para_prender = np.random.choice(indices_apagados, num_prender_real, replace=False)
            patron_distorsionado[indices_para_prender] = 1
            indices_ya_modificados.update(indices_para_prender)
            cambios_realizados += num_prender_real
        
        # 3c. Intercambiar píxeles (1↔0) en posiciones no modificadas aún
        if num_intercambiar > 0:
            # Filtrar índices que no fueron modificados en pasos anteriores
            indices_disponibles = [idx for idx in indices_totales if idx not in indices_ya_modificados]
            if len(indices_disponibles) > 0:
                num_intercambiar_real = min(num_intercambiar, len(indices_disponibles))
                indices_para_intercambiar = np.random.choice(indices_disponibles, num_intercambiar_real, replace=False)
                patron_distorsionado[indices_para_intercambiar] = 1 - patron_distorsionado[indices_para_intercambiar]
                cambios_realizados += num_intercambiar_real
        
        return patron_distorsionado
    
    def generar_dataset_equilibrado(self, cant, min_distorsion=1, max_distorsion=30, metodo_v2=False, modo_distorsion='mixto'):
        """
        Genera dataset equilibrado con distribución exacta.
        
        Args:
            cant: Cantidad total de ejemplos a generar
            min_distorsion: Porcentaje mínimo de distorsión sobre 100 píxeles totales (default: 1)
            max_distorsion: Porcentaje máximo de distorsión sobre 100 píxeles totales (default: 30)
            metodo_v2: (Ignorado, mantenido para compatibilidad)
            modo_distorsion: (Ignorado, siempre usa modo mixto: apagar/prender/intercambiar)
        
        Nota: La distorsión ahora se calcula sobre los 100 píxeles totales.
              Ejemplo: 20% distorsión = 20 cambios de píxeles (no sobre píxeles activos)
        """
        ejemplos_por_letra = cant // 3
        resto = cant % 3
        
        cant_B = ejemplos_por_letra
        cant_D = ejemplos_por_letra + (1 if resto >= 2 else 0)
        cant_F = ejemplos_por_letra + (1 if resto >= 1 else 0)
        
        def calcular_split(total):
            # Usar math.ceil() para garantizar MÍNIMO 10% (redondeo hacia arriba)
            perfectos = math.ceil(total * 0.10)
            distorsionados = total - perfectos
            return perfectos, distorsionados
        
        perfectos_B, dist_B = calcular_split(cant_B)
        perfectos_D, dist_D = calcular_split(cant_D)
        perfectos_F, dist_F = calcular_split(cant_F)
        
        patron_B = self.generar_letra('B')
        patron_D = self.generar_letra('D')
        patron_F = self.generar_letra('F')
        
        dataset = []
        
        for i in range(cant_B):
            if i < perfectos_B:
                fila = np.concatenate([patron_B, self.c_letras['B']])
            else:
                distorsion = np.random.uniform(min_distorsion, max_distorsion) / 100
                patron_dist = self.aplicar_distorsion(patron_B, distorsion, modo=modo_distorsion)
                fila = np.concatenate([patron_dist, self.c_letras['B']])
            dataset.append(fila)
        
        for i in range(cant_D):
            if i < perfectos_D:
                fila = np.concatenate([patron_D, self.c_letras['D']])
            else:
                distorsion = np.random.uniform(min_distorsion, max_distorsion) / 100
                patron_dist = self.aplicar_distorsion(patron_D, distorsion, modo=modo_distorsion)
                fila = np.concatenate([patron_dist, self.c_letras['D']])
            dataset.append(fila)
        
        for i in range(cant_F):
            if i < perfectos_F:
                fila = np.concatenate([patron_F, self.c_letras['F']])
            else:
                distorsion = np.random.uniform(min_distorsion, max_distorsion) / 100
                patron_dist = self.aplicar_distorsion(patron_F, distorsion, modo=modo_distorsion)
                fila = np.concatenate([patron_dist, self.c_letras['F']])
            dataset.append(fila)
        
        shuffle(dataset)
        
        dataframe = pd.DataFrame(dataset)
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        dataframe.to_csv(file_path, sep=";", index=None, header=None)
        
        total_perfectos = perfectos_B + perfectos_D + perfectos_F
        total_distorsionados = dist_B + dist_D + dist_F
        pct_perfectos = (total_perfectos / cant) * 100
        pct_distorsionados = (total_distorsionados / cant) * 100
        
        print(f"✅ Dataset de {cant} ejemplos generado (modo mixto)")
        print(f"   🎯 Distorsión calculada sobre 100 píxeles totales (no sobre píxeles activos)")
        print(f"   📊 Perfectos: {total_perfectos} ({pct_perfectos:.1f}%) - Mínimo garantizado: 10%")
        print(f"   🔀 Distorsionados: {total_distorsionados} ({pct_distorsionados:.1f}%)")
        print(f"   📏 Rango de distorsión: {min_distorsion}%-{max_distorsion}% = {min_distorsion} a {max_distorsion} cambios de píxeles")
    
    def cargar_dataset(self, cant):
        """
        Carga dataset desde CSV y separa patrones y etiquetas.
        
        Args:
            cant: Cantidad de ejemplos del dataset a cargar
            
        Returns:
            tuple: (X, y) donde X son los patrones y y las etiquetas one-hot
        """
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        df = pd.read_csv(file_path, sep=';', header=None)
        X = df.iloc[:, :100].values
        y = df.iloc[:, 100:].values
        return X, y
    
    def verificar_distribucion(self, cant):
        """
        Verifica y muestra la distribución del dataset por letra.
        
        Args:
            cant: Cantidad de ejemplos del dataset a verificar
            
        Returns:
            dict: Diccionario con conteos por letra {'B': count_b, 'D': count_d, 'F': count_f}
        """
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        df = pd.read_csv(file_path, sep=';', header=None)
        etiquetas = df.iloc[:, 100:].values
        
        # Contar por letra
        count_b = np.sum(np.all(etiquetas == [1, 0, 0], axis=1))
        count_d = np.sum(np.all(etiquetas == [0, 1, 0], axis=1))
        count_f = np.sum(np.all(etiquetas == [0, 0, 1], axis=1))
        
        print(f"📊 Verificación de distribución del dataset:")
        print(f"   Total ejemplos: {len(df)}")
        print(f"   B: {count_b} ejemplos ({count_b/len(df)*100:.1f}%)")
        print(f"   D: {count_d} ejemplos ({count_d/len(df)*100:.1f}%)")
        print(f"   F: {count_f} ejemplos ({count_f/len(df)*100:.1f}%)")
        print(f"\n✅ Distribución balanceada: {count_b} + {count_d} + {count_f} = {count_b+count_d+count_f}")
        
        return {'B': count_b, 'D': count_d, 'F': count_f}
