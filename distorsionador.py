"""
Clase Distorsionador para aplicar distorsión a patrones de letras.
"""

import random
import math
import numpy as np


class Distorsionador:
    """
    Clase para aplicar distorsión controlada a patrones de letras.
    
    Características:
    - Mantiene el 10% de las letras sin distorsión
    - Aplica distorsión aleatoria entre min y max
    - Intercambia posiciones de 1s por 0s (más realista)
    """
    
    def __init__(self, min_distorsion, max_distorsion):
        """
        Inicializa el distorsionador.
        
        Args:
            min_distorsion: Distorsión mínima (0.0 a 1.0, ej: 0.01 = 1%)
            max_distorsion: Distorsión máxima (0.0 a 1.0, ej: 0.30 = 30%)
        
        Ejemplos:
            # Distorsión entre 1% y 30%
            dist = Distorsionador(0.01, 0.30)
            
            # Distorsión entre 5% y 15%
            dist = Distorsionador(0.05, 0.15)
        """
        self.min_distorsion = min_distorsion
        self.max_distorsion = max_distorsion
    
    def _calcDistorsion(self):
        """
        Calcula la distorsión aleatoria en el rango especificado.
        
        Returns:
            float: Distorsión calculada (0.0 a 1.0)
        """
        distorsion = round(random.uniform(self.min_distorsion, self.max_distorsion), 2)
        return distorsion
    
    def distorsionar(self, letras):
        """
        Distorsiona un conjunto de letras manteniendo el 10% sin distorsión.
        
        Args:
            letras: Array 2D de letras (n_letras, 103) donde:
                    - [0:100] es el patrón de la letra
                    - [100:103] es la etiqueta one-hot
        
        Returns:
            Array 2D con letras distorsionadas
        """
        # El 10% de las letras tienen que mantenerse originales
        cant_sin_dist = int(0.1 * len(letras))
        letras_dist = letras.copy()
        
        for i in range(len(letras)):
            # Arranco a distorsionar desde la posición igual al 10% de la cantidad de letras
            if i >= cant_sin_dist:
                letra = letras[i].copy()
                distorsion = self._calcDistorsion()
                letras_dist[i] = self._dist_letra(letra, distorsion)
        
        return letras_dist

    def _dist_letra(self, letra, distorsion):
        """
        Distorsiona una letra individual convirtiendo 1s en 0s.
        
        Método más realista: degrada píxeles activos (1→0)
        en lugar de simplemente invertir píxeles aleatorios.
        
        Args:
            letra: Array 1D de 103 elementos (100 patrón + 3 etiqueta)
            distorsion: Porcentaje de distorsión (0.0 a 1.0)
        
        Returns:
            Array 1D con la letra distorsionada
        """
        # Recorro el array de la letra solo hasta la posición 100
        # (a partir de 100 están las etiquetas de clase)
        posiciones_uno = []
        
        for i in range(100):  # Solo los primeros 100 elementos (patrón)
            if letra[i] == 1:
                posiciones_uno.append(i)

        # Calcular cuántas celdas apagar según la distorsión
        celdas_apagar = math.ceil(distorsion * len(posiciones_uno))
        
        # Asegurarse de no exceder las posiciones disponibles
        celdas_apagar = min(celdas_apagar, len(posiciones_uno))

        # Apagar 1s (convertir a 0)
        posiciones_elegidas = random.sample(posiciones_uno, celdas_apagar)
        for posicion in posiciones_elegidas:
            letra[posicion] = 0
            
        return letra


if __name__ == "__main__":
    # Ejemplo de uso
    print("Ejemplo de uso de la clase Distorsionador\n")
    print("="*60)
    
    # Crear distorsionador
    distorsionador = Distorsionador(min_distorsion=0.01, max_distorsion=0.30)
    print(f"✅ Distorsionador creado:")
    print(f"   - Distorsión mínima: {distorsionador.min_distorsion*100}%")
    print(f"   - Distorsión máxima: {distorsionador.max_distorsion*100}%")
    
    # Crear datos de ejemplo (10 letras)
    print(f"\n📝 Creando 10 letras de ejemplo...")
    letras_ejemplo = []
    for i in range(10):
        # Patrón de ejemplo: 30 unos, 70 ceros + etiqueta [1,0,0]
        patron = np.concatenate([
            np.ones(30),
            np.zeros(70),
            np.array([1, 0, 0])  # Etiqueta
        ])
        letras_ejemplo.append(patron)
    
    letras_ejemplo = np.array(letras_ejemplo)
    print(f"   - Letras originales: {len(letras_ejemplo)}")
    print(f"   - Unos por letra (promedio): {np.mean([np.sum(l[:100]) for l in letras_ejemplo])}")
    
    # Distorsionar
    print(f"\n🔧 Aplicando distorsión...")
    letras_distorsionadas = distorsionador.distorsionar(letras_ejemplo)
    
    # Analizar resultados
    print(f"\n📊 Resultados:")
    letras_sin_cambios = 0
    for i in range(len(letras_ejemplo)):
        if np.array_equal(letras_ejemplo[i][:100], letras_distorsionadas[i][:100]):
            letras_sin_cambios += 1
    
    print(f"   - Letras sin distorsión: {letras_sin_cambios} ({letras_sin_cambios/len(letras_ejemplo)*100:.0f}%)")
    print(f"   - Letras con distorsión: {len(letras_ejemplo) - letras_sin_cambios}")
    print(f"   - Unos después de distorsión (promedio): {np.mean([np.sum(l[:100]) for l in letras_distorsionadas])}")
    
    # Verificar que las etiquetas no cambiaron
    etiquetas_iguales = all([np.array_equal(letras_ejemplo[i][100:], letras_distorsionadas[i][100:]) 
                             for i in range(len(letras_ejemplo))])
    print(f"   - Etiquetas preservadas: {'✅ Sí' if etiquetas_iguales else '❌ No'}")
    
    print("\n" + "="*60)
    print("✅ Ejemplo completado")
