from generador_dataset import GeneradorDataset

generador = GeneradorDataset()

print("🎯 Generando datasets con distorsiones específicas...\n")

# Ejemplo 1: Dataset de 500 con 5%, 13% y 20%
print("=" * 60)
print("Ejemplo 1: 500 ejemplos con distorsiones [5%, 13%, 20%]")
print("=" * 60)
generador.generar_data_con_distorsiones_especificas(
    cant=500,
    distorsiones=[1, 3, 5]
)

print("\n" + "=" * 60)
print("Ejemplo 2: 100 ejemplos con distorsiones [0%, 5%, 10%, 15%, 20%, 25%, 30%]")
print("=" * 60)
# Ejemplo 2: Dataset de 100 con rango de 5 en 5
generador.generar_data_con_distorsiones_especificas(
    cant=100,
    distorsiones=[0, 5, 10, 15, 20, 25, 30]
)

print("\n" + "=" * 60)
print("Ejemplo 3: 1000 ejemplos con distorsiones [0%, 10%, 20%, 30%]")
print("=" * 60)
# Ejemplo 3: Dataset de 1000 con valores específicos
generador.generar_data_con_distorsiones_especificas(
    cant=1000,
    distorsiones=[0, 10, 20, 30]  # 250 ejemplos de cada uno
)

print("\n" + "=" * 60)
print("✅ ¡Todos los datasets generados!")
print("=" * 60)