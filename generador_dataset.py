import os
import numpy as np
import pandas as pd
import csv
from random import shuffle

class GeneradorDataset:
    """
    Genera datasets de letras (B, D, F) con distorsión para entrenar MLP.
    Similar al código de referencia pero mejorado.
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
            for cant in ['100', '500', '1000']:
                path = os.path.join(self.base_path, 'data', tipo, cant)
                os.makedirs(path, exist_ok=True)
    
    def generar_letra(self, letra):
        """
        Genera una letra sin distorsión (patrón base).
        
        Args:
            letra: 'B', 'D' o 'F'
            
        Returns:
            Array 1D de 100 elementos
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
        
        letra_codigo = np.reshape(letra_codigo, (1, 100))[0]
        return letra_codigo
    
    def generar_data_letras(self, cant):
        """
        Genera dataset de letras originales (sin distorsión) y lo guarda en CSV.
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
        """
        letras = {
            "B": self.generar_letra("B"),
            "D": self.generar_letra("D"),
            "F": self.generar_letra("F")
        }
        
        tipo_letra = "B"
        letras_format_csv = []
        
        for i in range(int(cant)):
            if tipo_letra == "B":
                letras_format_csv.append(np.concatenate((letras["B"], self.c_letras["B"])))
                tipo_letra = "D"
            elif tipo_letra == "D":
                letras_format_csv.append(np.concatenate((letras["D"], self.c_letras["D"])))
                tipo_letra = "F"
            else:
                letras_format_csv.append(np.concatenate((letras["F"], self.c_letras["F"])))
                tipo_letra = "B"
        
        # Mezclar el dataset para más aleatoriedad
        shuffle(letras_format_csv)
        
        # Guardar en CSV
        file_path = os.path.join(self.base_path, "data", "originales", str(cant), 'letras.csv')
        file = open(file_path, 'w+', newline='')
        with file:
            write = csv.writer(file, delimiter=';')
            write.writerows(letras_format_csv)
        file.close()
        
        print(f"✅ Dataset original de {cant} ejemplos guardado en: {file_path}")
    
    def aplicar_distorsion(self, patron, porcentaje_distorsion):
        """
        Aplica distorsión aleatoria a un patrón.
        
        Args:
            patron: Array 1D de 100 elementos
            porcentaje_distorsion: Porcentaje de píxeles a distorsionar (0.01-0.30)
            
        Returns:
            Patrón distorsionado
        """
        patron_distorsionado = patron.copy()
        num_pixeles = len(patron)
        num_cambios = int(num_pixeles * porcentaje_distorsion)
        
        # Seleccionar píxeles aleatorios para distorsionar
        indices_cambiar = np.random.choice(num_pixeles, num_cambios, replace=False)
        
        # Invertir los valores (0 -> 1, 1 -> 0)
        for idx in indices_cambiar:
            patron_distorsionado[idx] = 1 - patron_distorsionado[idx]
        
        return patron_distorsionado
    
    def generar_data_distorsionadas(self, cant, porcentaje_sin_distorsion=0.10, 
                                   min_distorsion=0.01, max_distorsion=0.30):
        """
        Genera dataset con distorsión a partir de letras originales.
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
            porcentaje_sin_distorsion: % de ejemplos sin distorsión (default 10%)
            min_distorsion: Distorsión mínima (default 1%)
            max_distorsion: Distorsión máxima (default 30%)
        """
        # Leer letras originales
        letras_originales = self.get_letras_originales(cant)
        
        num_sin_distorsion = int(len(letras_originales) * porcentaje_sin_distorsion)
        letras_distorsionadas = []
        
        for i, fila in enumerate(letras_originales):
            patron = fila[:100]  # Primeros 100 elementos (letra)
            etiqueta = fila[100:]  # Últimos 3 elementos (clase)
            
            if i < num_sin_distorsion:
                # Sin distorsión
                letras_distorsionadas.append(np.concatenate((patron, etiqueta)))
            else:
                # Con distorsión aleatoria
                distorsion = np.random.uniform(min_distorsion, max_distorsion)
                patron_dist = self.aplicar_distorsion(patron, distorsion)
                letras_distorsionadas.append(np.concatenate((patron_dist, etiqueta)))
        
        # Mezclar
        shuffle(letras_distorsionadas)
        
        # Guardar en CSV
        dataframe_dist = pd.DataFrame(letras_distorsionadas)
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        dataframe_dist.to_csv(file_path, sep=";", index=None, header=None)
        
        print(f"✅ Dataset distorsionado de {cant} ejemplos guardado en: {file_path}")
        print(f"   - Sin distorsión: {num_sin_distorsion} ({porcentaje_sin_distorsion*100}%)")
        print(f"   - Con distorsión: {len(letras_originales) - num_sin_distorsion}")
        print(f"   - Rango distorsión: {min_distorsion*100}% - {max_distorsion*100}%")
    
    def get_letras_originales(self, cant):
        """
        Lee letras originales desde CSV.
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
            
        Returns:
            Array numpy con las letras
        """
        file_path = os.path.join(self.base_path, "data", "originales", str(cant), 'letras.csv')
        letras = pd.read_csv(file_path, sep=';', header=None).to_numpy()
        return letras
    
    def get_letras_distorsionadas(self, cant):
        """
        Lee letras distorsionadas desde CSV.
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
            
        Returns:
            Array numpy con las letras
        """
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        letras = pd.read_csv(file_path, sep=';', header=None).to_numpy()
        return letras
    
    def generar_data_con_distorsiones_especificas(self, cant, distorsiones=[0, 5, 10, 15, 20, 25, 30], mezclar=False):
        """
        Genera dataset con distorsiones específicas (no aleatorias).
        
        Args:
            cant: Cantidad TOTAL de ejemplos a generar
            distorsiones: Lista de % de distorsión a aplicar (ej: [0, 5, 10, 20, 30])
            mezclar: Si es True, mezcla los datos. Si es False, mantiene el orden para comparación
        
        Ejemplo:
            generador.generar_data_con_distorsiones_especificas(
                cant=700, 
                distorsiones=[0, 5, 13, 20],  # 175 ejemplos de cada distorsión
                mezclar=False  # Mantener orden para comparar con originales
            )
        """
        # Leer letras originales
        letras_originales = self.get_letras_originales(cant)
        
        # Distribuir ejemplos entre las distorsiones de forma equitativa
        ejemplos_por_distorsion = [cant // len(distorsiones)] * len(distorsiones)
        # Distribuir el resto entre las primeras distorsiones
        resto = cant % len(distorsiones)
        for i in range(resto):
            ejemplos_por_distorsion[i] += 1
        
        letras_con_distorsion = []
        idx = 0
        
        for i, distorsion_pct in enumerate(distorsiones):
            for _ in range(ejemplos_por_distorsion[i]):
                if idx >= len(letras_originales):
                    break
                    
                fila = letras_originales[idx]
                patron = fila[:100]  # Primeros 100 elementos (letra)
                etiqueta = fila[100:]  # Últimos 3 elementos (clase)
                
                # Aplicar distorsión específica
                if distorsion_pct == 0:
                    patron_dist = patron
                else:
                    patron_dist = self.aplicar_distorsion(patron, distorsion_pct / 100.0)
                
                letras_con_distorsion.append(np.concatenate((patron_dist, etiqueta)))
                idx += 1
        
        # Mezclar solo si se especifica
        if mezclar:
            shuffle(letras_con_distorsion)
            print("   🔀 Datos mezclados")
        else:
            print("   📌 Datos en orden (sin mezclar) para comparación")
        
        # Guardar en CSV
        dataframe_dist = pd.DataFrame(letras_con_distorsion)
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        dataframe_dist.to_csv(file_path, sep=";", index=None, header=None)
        
        print(f"✅ Dataset con distorsiones específicas guardado en: {file_path}")
        print(f"   - Total ejemplos: {len(letras_con_distorsion)}")
        print(f"   - Distribución: {dict(zip(distorsiones, ejemplos_por_distorsion))}")
        print(f"   - Distorsiones aplicadas: {distorsiones}%")
        
        return letras_con_distorsion
    
    def generar_todos_los_datasets(self):
        """Genera todos los datasets (100, 500, 1000) originales y distorsionados"""
        print("🔤 Generando todos los datasets...\n")
        
        for cant in [100, 500, 1000]:
            print(f"\n{'='*60}")
            print(f"📦 Generando dataset de {cant} ejemplos")
            print(f"{'='*60}")
            
            # Generar originales
            self.generar_data_letras(cant)
            
            # Generar distorsionados
            self.generar_data_distorsionadas(cant)
        
        print(f"\n{'='*60}")
        print("✅ Todos los datasets generados correctamente!")
        print(f"{'='*60}")

if __name__ == "__main__":
    generador = GeneradorDataset()
    generador.generar_todos_los_datasets()
