import os
import numpy as np
import pandas as pd
import csv
from random import shuffle
from distorsionador import Distorsionador

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
        self.distorsionador = None  # Se inicializa cuando sea necesario
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
    
    def generar_data_distorsionadas_v2(self, cant, min_distorsion=1.0, max_distorsion=30.0):
        """
        Genera dataset con distorsión usando la clase Distorsionador (intercambio 1s→0s).
        A diferencia de aplicar_distorsion que invierte aleatoriamente (0↔1),
        Distorsionador solo cambia 1s por 0s (más realista para degradación).
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
            min_distorsion: Distorsión mínima en % (default 1.0)
            max_distorsion: Distorsión máxima en % (default 30.0)
        """
        # Crear/actualizar distorsionador con nuevos parámetros
        self.distorsionador = Distorsionador(min_distorsion, max_distorsion)
        
        # Leer letras originales
        letras_originales = self.get_letras_originales(cant)
        
        # Aplicar distorsión (Distorsionador mantiene automáticamente 10% sin distorsión)
        letras_distorsionadas = self.distorsionador.distorsionar(letras_originales)
        
        # Mezclar
        shuffle(letras_distorsionadas)
        
        # Guardar en CSV
        dataframe_dist = pd.DataFrame(letras_distorsionadas)
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        dataframe_dist.to_csv(file_path, sep=";", index=None, header=None)
        
        num_sin_distorsion = int(len(letras_originales) * 0.10)
        print(f"✅ Dataset distorsionado (v2) de {cant} ejemplos guardado en: {file_path}")
        print(f"   - Sin distorsión: {num_sin_distorsion} (10%)")
        print(f"   - Con distorsión: {len(letras_originales) - num_sin_distorsion}")
        print(f"   - Rango distorsión: {min_distorsion}% - {max_distorsion}%")
        print(f"   - Método: Intercambio 1s→0s (Distorsionador)")
    
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
    
    def generar_data_con_distorsiones_especificas(self, cant, distorsion, mezclar=False):
        """
        Genera dataset con un porcentaje de distorsión específico.
        
        Args:
            cant: Cantidad TOTAL de ejemplos a generar
            distorsion: Porcentaje de distorsión (1-30)
            mezclar: Si es True, mezcla los datos. Si es False, mantiene el orden para comparación
        
        Raises:
            ValueError: Si la distorsión no está en el rango 1-30
        
        Ejemplo:
            generador.generar_data_con_distorsiones_especificas(
                cant=500, 
                distorsion=10,  # 10% de distorsión
                mezclar=False  # Mantener orden para comparar con originales
            )
        """
        # Validar que distorsion sea un número válido
        if not isinstance(distorsion, (int, float)):
            raise TypeError(f"La distorsión debe ser un número, no {type(distorsion).__name__}")
        
        if distorsion < 1 or distorsion > 30:
            raise ValueError(f"La distorsión debe estar entre 1 y 30. Recibido: {distorsion}")
        
        # Leer letras originales
        letras_originales = self.get_letras_originales(cant)
        
        letras_con_distorsion = []
        
        for idx in range(len(letras_originales)):
            fila = letras_originales[idx]
            patron = fila[:100]  # Primeros 100 elementos (letra)
            etiqueta = fila[100:]  # Últimos 3 elementos (clase)
            
            # Aplicar distorsión
            patron_dist = self.aplicar_distorsion(patron, distorsion / 100.0)
            letras_con_distorsion.append(np.concatenate((patron_dist, etiqueta)))
        
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
        
        print(f"✅ Dataset con distorsión de {distorsion}% guardado en: {file_path}")
        print(f"   - Total ejemplos: {len(letras_con_distorsion)}")
        print(f"   - Distorsión aplicada: {distorsion}%")
        
        return letras_con_distorsion
    
    def generar_todos_los_datasets(self, usar_distorsionador_v2=False):
        """
        Genera todos los datasets (100, 500, 1000) originales y distorsionados.
        
        Args:
            usar_distorsionador_v2: Si True, usa Distorsionador (1s→0s).
                                   Si False, usa método original (inversión aleatoria)
        """
        metodo = "Distorsionador (1s→0s)" if usar_distorsionador_v2 else "Inversión aleatoria (0↔1)"
        print(f"🔤 Generando todos los datasets...")
        print(f"   Método de distorsión: {metodo}\n")
        
        for cant in [100, 500, 1000]:
            print(f"\n{'='*60}")
            print(f"📦 Generando dataset de {cant} ejemplos")
            print(f"{'='*60}")
            
            # Generar originales
            self.generar_data_letras(cant)
            
            # Generar distorsionados
            if usar_distorsionador_v2:
                self.generar_data_distorsionadas_v2(cant)
            else:
                self.generar_data_distorsionadas(cant)
        
        print(f"\n{'='*60}")
        print("✅ Todos los datasets generados correctamente!")
        print(f"{'='*60}")

if __name__ == "__main__":
    generador = GeneradorDataset()
    
    print("=" * 70)
    print("GENERADOR DE DATASETS - Opciones de distorsión")
    print("=" * 70)
    print("\n1. Método CLÁSICO: Inversión aleatoria (0↔1)")
    print("   - Invierte píxeles aleatoriamente")
    print("   - Puede convertir 0→1 o 1→0")
    print("\n2. Método DISTORSIONADOR: Intercambio inteligente (1s→0s)")
    print("   - Solo cambia 1s por 0s")
    print("   - Más realista para degradación visual")
    print("   - Mantiene automáticamente 10% sin distorsión")
    print("=" * 70)
    
    opcion = input("\n¿Qué método deseas usar? (1/2) [default=1]: ").strip()
    
    usar_v2 = (opcion == "2")
    generador.generar_todos_los_datasets(usar_distorsionador_v2=usar_v2)
