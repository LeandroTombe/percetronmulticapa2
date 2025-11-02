import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
from dataset_generator import DatasetGenerator
import pickle
import os

class ClasificadorLetras:
    """
    Clasifica patrones de letras (B, D, F) distorsionados hasta 30%.
    """
    
    def __init__(self, mlp=None):
        """
        Args:
            mlp: Modelo MLP ya entrenado (opcional)
        """
        self.mlp = mlp
        self.generador = DatasetGenerator()
        self.letras_map = {0: 'B', 1: 'D', 2: 'F'}
        self.letras_map_inv = {'B': 0, 'D': 1, 'F': 2}
    
    def cargar_modelo(self, filepath):
        """
        Carga un modelo MLP previamente entrenado.
        
        Args:
            filepath: Ruta al archivo del modelo (.pkl)
        """
        with open(filepath, 'rb') as f:
            self.mlp = pickle.load(f)
        print(f"✅ Modelo cargado desde: {filepath}")
    
    def guardar_modelo(self, filepath):
        """
        Guarda el modelo MLP actual.
        
        Args:
            filepath: Ruta donde guardar el modelo (.pkl)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.mlp, f)
        print(f"✅ Modelo guardado en: {filepath}")
    
    def clasificar_patron(self, patron):
        """
        Clasifica un patrón (letra distorsionada).
        
        Args:
            patron: Array de 100 elementos (matriz 10x10 aplanada) o matriz 10x10
            
        Returns:
            dict con:
                - letra: 'B', 'D' o 'F'
                - clase: 0, 1, 2
                - probabilidades: array con las 3 probabilidades
                - confianza: probabilidad de la clase predicha
        """
        if self.mlp is None:
            raise ValueError("Debe cargar o entrenar un modelo primero")
        
        # Asegurar que el patrón sea 1D
        if patron.shape == (10, 10):
            patron = patron.flatten()
        
        # Predecir
        prediccion = self.mlp.predecir(patron.reshape(1, -1))[0]
        
        # Obtener clase predicha
        clase = np.argmax(prediccion)
        letra = self.letras_map[clase]
        confianza = prediccion[clase]
        
        return {
            'letra': letra,
            'clase': clase,
            'probabilidades': prediccion,
            'confianza': confianza
        }
    
    def clasificar_lote(self, patrones):
        """
        Clasifica múltiples patrones.
        
        Args:
            patrones: Array de shape (n, 100) o (n, 10, 10)
            
        Returns:
            Lista de diccionarios con resultados
        """
        if self.mlp is None:
            raise ValueError("Debe cargar o entrenar un modelo primero")
        
        # Asegurar forma correcta
        if patrones.ndim == 3:  # (n, 10, 10)
            patrones = patrones.reshape(len(patrones), -1)
        
        resultados = []
        for patron in patrones:
            resultado = self.clasificar_patron(patron)
            resultados.append(resultado)
        
        return resultados
    
    def generar_patron_distorsionado(self, letra, porcentaje_distorsion):
        """
        Genera un patrón de letra con distorsión específica.
        
        Args:
            letra: 'B', 'D' o 'F'
            porcentaje_distorsion: 0-30 (porcentaje de distorsión)
            
        Returns:
            Patrón distorsionado (array de 100 elementos)
        """
        patron_base = self.generador.generar_letra(letra)
        
        if porcentaje_distorsion > 0:
            patron_distorsionado = self.generador.aplicar_distorsion(
                patron_base, 
                porcentaje_distorsion / 100.0
            )
            return patron_distorsionado
        else:
            return patron_base
    
    def probar_distorsiones(self, letra, distorsiones=[0, 5, 10, 15, 20, 25, 30]):
        """
        Prueba clasificación con diferentes niveles de distorsión.
        
        Args:
            letra: 'B', 'D' o 'F'
            distorsiones: Lista de porcentajes de distorsión a probar
            
        Returns:
            DataFrame con resultados
        """
        import pandas as pd
        
        if self.mlp is None:
            raise ValueError("Debe cargar o entrenar un modelo primero")
        
        resultados = []
        
        print(f"\n🔍 Probando clasificación de letra '{letra}' con diferentes distorsiones:")
        print("=" * 70)
        
        for dist in distorsiones:
            # Generar patrón distorsionado
            patron = self.generar_patron_distorsionado(letra, dist)
            
            # Clasificar
            resultado = self.clasificar_patron(patron)
            
            # Verificar si es correcto
            correcto = resultado['letra'] == letra
            
            resultados.append({
                'Letra Real': letra,
                'Distorsión (%)': dist,
                'Predicción': resultado['letra'],
                'Confianza (%)': round(resultado['confianza'] * 100, 2),
                'Correcto': '✅' if correcto else '❌',
                'Prob B': round(resultado['probabilidades'][0], 3),
                'Prob D': round(resultado['probabilidades'][1], 3),
                'Prob F': round(resultado['probabilidades'][2], 3)
            })
            
            # Mostrar resultado
            status = '✅' if correcto else '❌'
            print(f"{status} Distorsión {dist:2d}% → Predicción: {resultado['letra']} "
                  f"(Confianza: {resultado['confianza']*100:.1f}%)")
        
        print("=" * 70)
        
        df = pd.DataFrame(resultados)
        return df
    
    def evaluar_robustez(self, num_pruebas=10, max_distorsion=30):
        """
        Evalúa la robustez del clasificador con múltiples pruebas aleatorias.
        
        Args:
            num_pruebas: Número de pruebas por letra y nivel de distorsión
            max_distorsion: Distorsión máxima a probar (%)
            
        Returns:
            DataFrame con estadísticas
        """
        import pandas as pd
        
        if self.mlp is None:
            raise ValueError("Debe cargar o entrenar un modelo primero")
        
        print(f"\n📊 Evaluando robustez con {num_pruebas} pruebas por configuración...")
        
        resultados = []
        distorsiones = range(0, max_distorsion + 1, 5)
        
        for letra in ['B', 'D', 'F']:
            for dist in distorsiones:
                correctos = 0
                confianzas = []
                
                for _ in range(num_pruebas):
                    patron = self.generar_patron_distorsionado(letra, dist)
                    resultado = self.clasificar_patron(patron)
                    
                    if resultado['letra'] == letra:
                        correctos += 1
                    confianzas.append(resultado['confianza'])
                
                precision = (correctos / num_pruebas) * 100
                confianza_promedio = np.mean(confianzas) * 100
                
                resultados.append({
                    'Letra': letra,
                    'Distorsión (%)': dist,
                    'Precisión (%)': round(precision, 1),
                    'Confianza Promedio (%)': round(confianza_promedio, 1),
                    'Correctos': f"{correctos}/{num_pruebas}"
                })
        
        df = pd.DataFrame(resultados)
        
        # Mostrar resumen
        print("\n📈 Resumen por nivel de distorsión:")
        print("=" * 60)
        for dist in distorsiones:
            df_dist = df[df['Distorsión (%)'] == dist]
            prec_promedio = df_dist['Precisión (%)'].mean()
            print(f"Distorsión {dist:2d}% → Precisión promedio: {prec_promedio:.1f}%")
        print("=" * 60)
        
        return df
    
    def visualizar_clasificacion(self, letra, distorsion=20):
        """
        Visualiza un ejemplo de clasificación con distorsión.
        
        Args:
            letra: 'B', 'D' o 'F'
            distorsion: Porcentaje de distorsión
        """
        if self.mlp is None:
            raise ValueError("Debe cargar o entrenar un modelo primero")
        
        # Generar patrones
        patron_original = self.generador.generar_letra(letra).reshape(10, 10)
        patron_distorsionado = self.generar_patron_distorsionado(letra, distorsion)
        patron_dist_2d = patron_distorsionado.reshape(10, 10)
        
        # Clasificar
        resultado = self.clasificar_patron(patron_distorsionado)
        
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
        
        return resultado
    
    def clasificar_interactivo(self):
        """
        Modo interactivo para clasificar patrones.
        """
        if self.mlp is None:
            raise ValueError("Debe cargar o entrenar un modelo primero")
        
        print("\n" + "="*60)
        print("🎯 CLASIFICADOR INTERACTIVO DE LETRAS")
        print("="*60)
        
        while True:
            print("\nOpciones:")
            print("1. Clasificar letra con distorsión")
            print("2. Probar diferentes distorsiones")
            print("3. Evaluar robustez")
            print("4. Salir")
            
            opcion = input("\nSeleccione una opción (1-4): ").strip()
            
            if opcion == '1':
                letra = input("Ingrese letra (B/D/F): ").strip().upper()
                if letra not in ['B', 'D', 'F']:
                    print("❌ Letra inválida")
                    continue
                
                try:
                    distorsion = int(input("Ingrese distorsión (0-30%): ").strip())
                    if distorsion < 0 or distorsion > 30:
                        print("❌ Distorsión debe estar entre 0 y 30")
                        continue
                except ValueError:
                    print("❌ Ingrese un número válido")
                    continue
                
                self.visualizar_clasificacion(letra, distorsion)
            
            elif opcion == '2':
                letra = input("Ingrese letra (B/D/F): ").strip().upper()
                if letra not in ['B', 'D', 'F']:
                    print("❌ Letra inválida")
                    continue
                
                df = self.probar_distorsiones(letra)
                print("\n📊 Tabla de resultados:")
                print(df.to_string(index=False))
            
            elif opcion == '3':
                try:
                    num_pruebas = int(input("Número de pruebas por configuración (default 10): ").strip() or "10")
                except ValueError:
                    num_pruebas = 10
                
                df = self.evaluar_robustez(num_pruebas=num_pruebas)
                print("\n📊 Tabla completa:")
                print(df.to_string(index=False))
            
            elif opcion == '4':
                print("\n👋 ¡Hasta luego!")
                break
            
            else:
                print("❌ Opción inválida")


# ============================================
# EJEMPLO DE USO
# ============================================
if __name__ == "__main__":
    print("🎯 EJEMPLO: Clasificador de Letras Distorsionadas\n")
    
    # 1. Crear y entrenar un modelo simple
    print("1️⃣ Creando y entrenando modelo...")
    from dataset_generator import DatasetGenerator
    
    generador = DatasetGenerator()
    
    # Generar datos si no existen
    if not os.path.exists('data/distorsionadas/100/letras.csv'):
        print("   Generando datasets...")
        generador.generar_todos_los_datasets()
    
    # Cargar dataset
    datos = generador.get_letras_distorsionadas(500)
    X = datos[:, :100]  # Patrones
    y_labels = datos[:, 100:]  # Etiquetas one-hot
    
    # Crear y entrenar MLP
    mlp = MLP(
        arquitectura=[100, 8, 3],
        funciones_activacion=['sigmoidal', 'lineal'],
        learning_rate=0.1,
        momentum=0.5
    )
    
    print("   Entrenando...")
    mlp.entrenar(X, y_labels, epochs=200, verbose=False)
    print("   ✅ Modelo entrenado\n")
    
    # 2. Crear clasificador
    clasificador = ClasificadorLetras(mlp)
    
    # 3. Guardar modelo
    clasificador.guardar_modelo('modelos/mlp_letras.pkl')
    
    # 4. Ejemplo: Clasificar letra con distorsión
    print("2️⃣ Ejemplo: Clasificar letra 'B' con 20% de distorsión")
    resultado = clasificador.visualizar_clasificacion('B', distorsion=20)
    
    # 5. Probar diferentes distorsiones
    print("\n3️⃣ Probando diferentes niveles de distorsión en letra 'D':")
    df_dist = clasificador.probar_distorsiones('D')
    print(df_dist.to_string(index=False))
    
    # 6. Evaluar robustez completa
    print("\n4️⃣ Evaluando robustez del clasificador:")
    df_robustez = clasificador.evaluar_robustez(num_pruebas=5)
    print(df_robustez.to_string(index=False))
    
    # 7. Modo interactivo (descomentar para usar)
    # clasificador.clasificar_interactivo()
    
    print("\n✅ Ejemplos completados!")
