import os
import numpy as np
import pandas as pd
import csv
import math
import random
from random import shuffle

# Import opcional de Distorsionador
try:
    from distorsionador import Distorsionador
except ImportError:
    Distorsionador = None  # Si no existe, los métodos v2 no estarán disponibles

class GeneradorDataset:
    """
    Genera datasets de letras (B, D, F) con distorsión para entrenar MLP.
    Similar al código de referencia pero mejorado.
    
    REPRODUCIBILIDAD:
    ================
    Todos los métodos usan semillas fijas para garantizar que los datasets
    generados sean siempre idénticos, permitiendo comparaciones justas y
    análisis reproducibles.
    
    Semillas utilizadas:
    - numpy: 42 (para distorsiones y selecciones aleatorias)
    - random: 42 (para shuffle y random.uniform)
    """
    
    def __init__(self, seed=42):
        """
        Inicializa el generador de datasets con semilla fija para reproducibilidad.
        
        Args:
            seed: Semilla para generadores aleatorios (default: 42)
                  Usar la misma semilla garantiza datasets idénticos
        """
        self.seed = seed
        self.letras = ['B', 'D', 'F']
        self.c_letras = {
            'B': np.array([1, 0, 0]),
            'D': np.array([0, 1, 0]),
            'F': np.array([0, 0, 1])
        }
        self.base_path = os.path.abspath('')
        self.distorsionador = None  # Se inicializa cuando sea necesario
        
        # Configurar semillas para reproducibilidad
        self._configurar_semillas()
        self._crear_directorios()
    
    def _configurar_semillas(self):
        """
        Configura las semillas de todos los generadores aleatorios.
        Esto garantiza que los datasets sean reproducibles.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        print(f"🌱 Semillas configuradas: seed={self.seed} (datasets reproducibles)")
    
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
        
        REPRODUCIBLE: Usa semilla fija para shuffle.
        
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
        
        # Mezclar el dataset con semilla fija para reproducibilidad
        random.seed(self.seed)
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
        
        REPRODUCIBLE: Usa np.random con semilla configurada en __init__.
        
        Args:
            patron: Array 1D de 100 elementos
            porcentaje_distorsion: Porcentaje de píxeles a distorsionar (0.01-0.30)
            
        Returns:
            Patrón distorsionado
        """
        patron_distorsionado = patron.copy()
        num_pixeles = len(patron)
        num_cambios = int(num_pixeles * porcentaje_distorsion)
        
        # Seleccionar píxeles aleatorios para distorsionar (usa semilla global)
        indices_cambiar = np.random.choice(num_pixeles, num_cambios, replace=False)
        
        # Invertir los valores (0 -> 1, 1 -> 0)
        for idx in indices_cambiar:
            patron_distorsionado[idx] = 1 - patron_distorsionado[idx]
        
        return patron_distorsionado
    
    def generar_data_distorsionadas(self, cant, porcentaje_sin_distorsion=0.10, 
                                   min_distorsion=0.01, max_distorsion=0.30):
        """
        Genera dataset con distorsión a partir de letras originales.
        
        REPRODUCIBLE: Usa semillas configuradas para distorsiones y shuffle.
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
            porcentaje_sin_distorsion: % de ejemplos sin distorsión (default 10%)
            min_distorsion: Distorsión mínima (default 1%)
            max_distorsion: Distorsión máxima (default 30%)
        """
        # Reiniciar semilla para garantizar reproducibilidad
        np.random.seed(self.seed)
        random.seed(self.seed)
        
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
                # Con distorsión aleatoria (reproducible)
                distorsion = np.random.uniform(min_distorsion, max_distorsion)
                patron_dist = self.aplicar_distorsion(patron, distorsion)
                letras_distorsionadas.append(np.concatenate((patron_dist, etiqueta)))
        
        # Mezclar con semilla para reproducibilidad
        random.seed(self.seed)
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
        
        REPRODUCIBLE: Usa semilla fija para distorsiones y shuffle.
        
        Args:
            cant: Cantidad de ejemplos (100, 500, 1000)
            min_distorsion: Distorsión mínima en % (default 1.0)
            max_distorsion: Distorsión máxima en % (default 30.0)
        """
        # Reiniciar semilla para garantizar reproducibilidad
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Crear/actualizar distorsionador con nuevos parámetros
        self.distorsionador = Distorsionador(min_distorsion, max_distorsion)
        
        # Leer letras originales
        letras_originales = self.get_letras_originales(cant)
        
        # Aplicar distorsión (Distorsionador mantiene automáticamente 10% sin distorsión)
        letras_distorsionadas = self.distorsionador.distorsionar(letras_originales)
        
        # Mezclar con semilla para reproducibilidad
        random.seed(self.seed)
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
        
        REPRODUCIBLE: Usa semilla fija para distorsiones y shuffle opcional.
        
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
        # Reiniciar semilla para garantizar reproducibilidad
        np.random.seed(self.seed)
        random.seed(self.seed)
        
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
            
            # Aplicar distorsión (reproducible)
            patron_dist = self.aplicar_distorsion(patron, distorsion / 100.0)
            letras_con_distorsion.append(np.concatenate((patron_dist, etiqueta)))
        
        # Mezclar solo si se especifica (con semilla para reproducibilidad)
        if mezclar:
            random.seed(self.seed)
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
    
    def generar_dataset_equilibrado(self, cant, min_distorsion=1, max_distorsion=30, metodo_v2=False):
        """
        Genera un dataset equilibrado según la especificación del usuario:
        
        Distribución exacta:
        - cant=100: 10 perfectos (3B+3D+4F) y 90 distorsionados (30B+30D+30F)
        - cant=500: 50 perfectos (16B+17D+17F) y 450 distorsionados (150B+150D+150F)
        - cant=1000: 100 perfectos (33B+33D+34F) y 900 distorsionados (300B+300D+300F)
        
        REPRODUCIBLE: Usa semilla fija para todas las distorsiones y shuffle.
        Generará EXACTAMENTE el mismo dataset cada vez que se ejecute con los mismos parámetros.
        
        Args:
            cant: int, número total de ejemplos (debe ser 100, 500 o 1000)
            min_distorsion: int, porcentaje mínimo de distorsión (1-30)
            max_distorsion: int, porcentaje máximo de distorsión (1-30)
            metodo_v2: bool, si True usa Distorsionador (apaga 1s), si False usa inversión 0↔1
        
        Retorna:
            Lista de filas generadas (lista de arrays 103-long) y guarda el CSV en
            data/distorsionadas/<cant>/letras.csv
        """
        # Reiniciar semilla al inicio para garantizar reproducibilidad total
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if cant not in (100, 500, 1000):
            raise ValueError("cant debe ser 100, 500 o 1000")
        
        # 1) Calcular cuántos ejemplos perfectos (10%) y distorsionados (90%)
        total_perfectos = int(cant * 0.10)
        total_distorsionados = cant - total_perfectos
        
        # 2) Distribuir perfectos entre las 3 letras (lo más equitativo posible)
        # Asignación: últimas letras reciben el resto (para que F reciba más cuando cant=100)
        perfectos_base = total_perfectos // 3
        perfectos_resto = total_perfectos % 3
        perfectos_por_letra = [perfectos_base, perfectos_base, perfectos_base]
        # Asignar desde el final: índices [2, 1, 0] si hay resto
        for i in range(perfectos_resto):
            perfectos_por_letra[2 - i] += 1
        
        # 3) Distribuir distorsionados entre las 3 letras (lo más equitativo posible)
        distorsionados_base = total_distorsionados // 3
        distorsionados_resto = total_distorsionados % 3
        distorsionados_por_letra = [distorsionados_base, distorsionados_base, distorsionados_base]
        # Igual lógica: últimas letras reciben el resto
        for i in range(distorsionados_resto):
            distorsionados_por_letra[2 - i] += 1
        
        # Patrones base y etiquetas
        letras_patron = {
            'B': self.generar_letra('B'),
            'D': self.generar_letra('D'),
            'F': self.generar_letra('F')
        }
        etiquetas = self.c_letras
        
        # 4) Construir filas: para cada letra añadir perfectos y distorsionados
        filas = []
        for idx, letra in enumerate(['B', 'D', 'F']):
            perfectos = perfectos_por_letra[idx]
            distorsionados = distorsionados_por_letra[idx]
            
            # Añadir perfectos
            for _ in range(perfectos):
                filas.append(np.concatenate((letras_patron[letra], etiquetas[letra])))
            
            # Añadir distorsionados
            for _ in range(distorsionados):
                patron = letras_patron[letra].copy()
                if metodo_v2:
                    # usar Distorsionador: apaga 1s→0s
                    fila_temp = np.concatenate((patron, etiquetas[letra]))
                    dist = round(random.uniform(min_distorsion, max_distorsion) / 100.0, 2)
                    d = Distorsionador(min_distorsion/100.0, max_distorsion/100.0)
                    fila_dist = d._dist_letra(fila_temp.copy(), dist)
                    filas.append(fila_dist)
                else:
                    # método clásico: invertir píxeles aleatorios 0↔1
                    dist = random.uniform(min_distorsion / 100.0, max_distorsion / 100.0)
                    patron_dist = self.aplicar_distorsion(patron, dist)
                    filas.append(np.concatenate((patron_dist, etiquetas[letra])))
        
        # 5) Mezclar con semilla y guardar
        random.seed(self.seed)
        random.shuffle(filas)
        dataframe = pd.DataFrame(filas)
        file_path = os.path.join(self.base_path, "data", "distorsionadas", str(cant), 'letras.csv')
        dataframe.to_csv(file_path, sep=';', index=None, header=None)
        
        print(f"✅ Dataset equilibrado de {cant} ejemplos guardado en: {file_path}")
        print(f"   📊 Distribución:")
        print(f"      Perfectos: {total_perfectos} total → B:{perfectos_por_letra[0]} D:{perfectos_por_letra[1]} F:{perfectos_por_letra[2]}")
        print(f"      Distorsionados: {total_distorsionados} total → B:{distorsionados_por_letra[0]} D:{distorsionados_por_letra[1]} F:{distorsionados_por_letra[2]}")
        print(f"   🎲 Distorsión: {min_distorsion}% - {max_distorsion}%")
        print(f"   🌱 Reproducible: seed={self.seed}")
        
        return filas
    
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

