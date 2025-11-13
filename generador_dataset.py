"""
Generador de datasets de letras (B, D, F) con distorsión para entrenar MLP.
"""

import numpy as np
import pandas as pd
import os
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
        """Crea las carpetas necesarias para guardar los datos"""
        for tipo in ['originales', 'distorsionadas']:
            for cant in [10, 20, 50, 80, 100, 500, 1000]:
                path = os.path.join(self.base_path, "data", tipo, str(cant))
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
        Aplica distorsión inteligente basada en los píxeles activos (1s) de la letra.
        
        Args:
            patron: Array 1D de 100 elementos
            porcentaje_distorsion: Porcentaje de píxeles ACTIVOS a distorsionar (0.01-0.30)
            modo: Tipo de distorsión a aplicar:
                  - 'apagar': Solo apaga píxeles (1 → 0)
                  - 'prender': Solo prende píxeles (0 → 1)
                  - 'intercambiar': Intercambia píxeles (1 ↔ 0)
                  - 'mixto': Combinación aleatoria de apagar y prender (default)
            
        Returns:
            Patrón distorsionado (copia, no modifica el original)
        """
        patron_distorsionado = patron.copy()
        
        # 1. Contar píxeles activos (1s) en el patrón original
        indices_activos = np.where(patron == 1)[0]
        num_pixeles_activos = len(indices_activos)
        
        # 2. Calcular cuántos píxeles activos distorsionar
        num_cambios = int(num_pixeles_activos * porcentaje_distorsion)
        
        if num_cambios == 0:
            return patron_distorsionado
        
        # 3. Aplicar distorsión según el modo
        if modo == 'apagar':
            # Solo apagar píxeles activos (1 → 0)
            indices_apagar = np.random.choice(indices_activos, num_cambios, replace=False)
            patron_distorsionado[indices_apagar] = 0
            
        elif modo == 'prender':
            # Solo prender píxeles apagados (0 → 1)
            indices_apagados = np.where(patron == 0)[0]
            if len(indices_apagados) > 0:
                num_prender = min(num_cambios, len(indices_apagados))
                indices_prender = np.random.choice(indices_apagados, num_prender, replace=False)
                patron_distorsionado[indices_prender] = 1
                
        elif modo == 'intercambiar':
            # Intercambiar píxeles aleatorios (1 ↔ 0)
            indices_totales = np.arange(len(patron))
            indices_cambiar = np.random.choice(indices_totales, num_cambios, replace=False)
            patron_distorsionado[indices_cambiar] = 1 - patron_distorsionado[indices_cambiar]
            
        else:  # modo == 'mixto' (default)
            # Combinación: mitad apagar, mitad prender
            num_apagar = num_cambios // 2
            num_prender = num_cambios - num_apagar
            
            # Apagar algunos píxeles activos
            if num_apagar > 0 and len(indices_activos) > 0:
                indices_apagar = np.random.choice(indices_activos, 
                                                  min(num_apagar, len(indices_activos)), 
                                                  replace=False)
                patron_distorsionado[indices_apagar] = 0
            
            # Prender algunos píxeles apagados
            indices_apagados = np.where(patron == 0)[0]
            if num_prender > 0 and len(indices_apagados) > 0:
                indices_prender = np.random.choice(indices_apagados, 
                                                   min(num_prender, len(indices_apagados)), 
                                                   replace=False)
                patron_distorsionado[indices_prender] = 1
        
        return patron_distorsionado
    
    def generar_dataset_equilibrado(self, cant, min_distorsion=1, max_distorsion=30, metodo_v2=False, modo_distorsion='mixto'):
        """
        Genera dataset equilibrado con distribución exacta.
        
        Args:
            cant: Cantidad total de ejemplos a generar
            min_distorsion: Porcentaje mínimo de distorsión (default: 1)
            max_distorsion: Porcentaje máximo de distorsión (default: 30)
            metodo_v2: (Ignorado, mantenido para compatibilidad)
            modo_distorsion: Tipo de distorsión ('apagar', 'prender', 'intercambiar', 'mixto')
        """
        ejemplos_por_letra = cant // 3
        resto = cant % 3
        
        cant_B = ejemplos_por_letra
        cant_D = ejemplos_por_letra + (1 if resto >= 2 else 0)
        cant_F = ejemplos_por_letra + (1 if resto >= 1 else 0)
        
        def calcular_split(total):
            perfectos = int(total * 0.10)
            distorsionados = total - perfectos
            return perfectos, distorsionados
        
        perfectos_B, dist_B = calcular_split(cant_B)
        perfectos_D, dist_D = calcular_split(cant_D)
        perfectos_F, dist_F = calcular_split(cant_F)
        
        patron_B = self.generar_letra('B')
        patron_D = self.generar_letra('D')
        patron_F = self.generar_letra('F')
        
        # Contar píxeles activos por letra (para información)
        pixeles_B = np.sum(patron_B == 1)
        pixeles_D = np.sum(patron_D == 1)
        pixeles_F = np.sum(patron_F == 1)
        
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
        
        print(f"✅ Dataset de {cant} ejemplos generado (modo: {modo_distorsion})")
        print(f"   🎯 Píxeles activos por letra: B={pixeles_B}, D={pixeles_D}, F={pixeles_F}")
        print(f"   📊 Perfectos: {perfectos_B + perfectos_D + perfectos_F} (10%)")
        print(f"   🔀 Distorsionados: {dist_B + dist_D + dist_F} (90%)")
    
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
