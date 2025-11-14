import anywidget
import traitlets
import os
import pandas as pd
import mlp
from generador_dataset import GeneradorDataset # Import GeneradorDataset
from sklearn.model_selection import train_test_split
from visualizador import graficar_mse_entrenamiento_validacion

def leer_dataset(cantidad, tipo='originales'):
    """Lee un dataset desde CSV"""
    file_path = os.path.join('data', tipo, str(cantidad), 'letras.csv')
    df = pd.read_csv(file_path, sep=';', header=None)
    X = df.iloc[:, :100].values
    y = df.iloc[:, 100:].values
    return X, y

class NeuralNetworkInterface(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
        // Estilos CSS
        const styles = `
            .main-container {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .section {
                background: #f9f9f9;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            .section-title {
                color: #333;
                margin-top: 0;
                margin-bottom: 20px;
                border-bottom: 2px solid #2196F3;
                padding-bottom: 10px;
            }
            .control-group {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin: 15px 0;
            }
            .control-item {
                display: flex;
                flex-direction: column;
                gap: 5px;
                min-width: 150px;
            }
            .control-label {
                font-weight: bold;
                color: #555;
                font-size: 14px;
            }
            .select-control {
                padding: 8px;
                border: 1px solid #ccc;
                background: white;
                border-radius: 4px;
                font-size: 14px;
                cursor: pointer;
                min-width: 120px;
                transition: all 0.2s;
            }
            .select-control:hover {
                border-color: #2196F3;
                box-shadow: 0 0 5px rgba(33, 150, 243, 0.3);
            }
            .select-control:focus {
                outline: none;
                border-color: #2196F3;
                box-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
            }
            .number-input {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                width: 100px;
                font-size: 14px;
            }
            .readonly-label {
                padding: 8px;
                background: #e8e8e8;
                border: 1px solid #ccc;
                border-radius: 4px;
                color: #333;
                font-weight: bold;
            }
            .matrix-container {
                display: grid;
                grid-template-columns: repeat(10, 30px);
                grid-template-rows: repeat(10, 30px);
                gap: 1px;
                padding: 10px;
                background-color: #333;
                border: 2px solid #333;
                border-radius: 5px;
                width: fit-content;
                margin: 20px auto;
            }
            .matrix-cell {
                width: 30px;
                height: 30px;
                background-color: white;
                border: 1px solid #ccc;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .matrix-cell:hover {
                background-color: #f0f0f0 !important;
            }
            .clear-button{
                margin-top: 10px;
                padding: 8px 16px;
                color: white;
                background-color: #f44336;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-family: Arial, sans-serif;
            },
            .create-button {
                margin-top: 10px;
                padding: 8px 16px;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-family: Arial, sans-serif;
            },
            .results-label {
                background: #e3f2fd;
                border: 2px solid #2196F3;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #1976D2;
                text-align: center;
                min-height: 20px;
                font-weight: bold;
            }
            .letter-selector {
                margin: 15px 0;
                text-align: center;
            }
            .letter-button {
                margin: 0 5px;
                padding: 10px 15px;
                border: 2px solid #2196F3;
                background: white;
                color: #2196F3;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.2s;
            }
            .letter-button:hover {
                background: #e3f2fd;
            }
            .letter-button.selected {
                background: #2196F3;
                color: white;
            }
        `;

        // Agregar estilos
        let styleSheet = document.createElement("style");
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);

        // Contenedor principal
        let mainContainer = document.createElement("div");
        mainContainer.className = "main-container";

        // Función para crear etiqueta de resultados
        function createDisplayInfo(initialText = "Resultados aparecerán aquí...", type = "info") {
            let displayInfo = document.createElement("div");
            displayInfo.className = "results-label";
            displayInfo.innerHTML = initialText;
            displayInfo.setAttribute("data-type", type);
            return displayInfo;
        }

        // Variable para almacenar la etiqueta de resultados del entrenamiento, declarada una sola vez aquí
        let trainingDisplayInfo = createDisplayInfo("Configuración de entrenamiento aparecerá aquí...", "info");

        // === SECCIÓN 1: CONFIGURACIÓN DE ENTRENAMIENTO ===
        let trainingSection = document.createElement("div");
        trainingSection.className = "section";

        let trainingTitle = document.createElement("h2");
        trainingTitle.className = "section-title";
        trainingTitle.innerHTML = "Configuración de Entrenamiento";
        trainingSection.appendChild(trainingTitle);

        // Variables para almacenar selecciones
        let selections = {
            training: {
                hiddenLayers: 1,
                neuronsPerLayer: 5,
                neuronsPerLayer2: 5,
                dataset: 100,
                validationPercent: 10,
                learningRate: 0.1, // Corrected initialization
                momentum: 0.5,
                epochs: 100
            }
        };

        // Función para crear select dropdown
        function createSelectControl(label, options, selectedValue, onSelect, id = null, disabled = null) {
            let controlItem = document.createElement("div");
            if (id){
              controlItem.id = id + "-item"; // Add -item to the container id
            }
            controlItem.className = "control-item";

            let labelEl = document.createElement("div");
            labelEl.className = "control-label";
            labelEl.innerHTML = label;
            controlItem.appendChild(labelEl);

            let select = document.createElement("select");
            select.className = "select-control";
            if (id){
              select.id = id; // Set id on the select element
            }

            if (disabled !== null){
              select.disabled = disabled;
            }

            options.forEach(option => {
                let optionEl = document.createElement("option");
                optionEl.value = option;
                optionEl.textContent = option;
                if (option === selectedValue) {
                    optionEl.selected = true;
                }
                select.appendChild(optionEl);
            });

            select.addEventListener("change", (e) => {
                let value = e.target.value;
                // Convertir a número si es posible
                if (!isNaN(value) && value !== '') {
                    value = parseFloat(value);
                }
                onSelect(value);
            });

            controlItem.appendChild(select);
            return controlItem;
        }

        // Función para crear input numérico
        function createNumberInput(label, value, onchange, min = null, max = null, step = null) {
            let controlItem = document.createElement("div");
            controlItem.className = "control-item";

            let labelEl = document.createElement("div");
            labelEl.className = "control-label";
            labelEl.innerHTML = label;
            controlItem.appendChild(labelEl);

            let input = document.createElement("input");
            input.type = "number";
            input.className = "number-input";
            input.value = value;
            if (min !== null) input.min = min;
            if (max !== null) input.max = max;
            if (step !== null) input.step = step;

            input.addEventListener("change", (e) => {
                let val = e.target.value;
                // Si step es null o 1, usar parseInt, sino parseFloat
                let parsedValue = (step === null || step === 1) ? parseInt(val) : parseFloat(val);
                onchange(parsedValue || (min !== null ? min : 0));
            });

            controlItem.appendChild(input);
            return controlItem;
        }

        // Función para crear label de solo lectura
        function createReadOnlyLabel(label, value) {
            let controlItem = document.createElement("div");
            controlItem.className = "control-item";

            let labelEl = document.createElement("div");
            labelEl.className = "control-label";
            labelEl.innerHTML = label;
            controlItem.appendChild(labelEl);

            let readOnlyEl = document.createElement("div");
            readOnlyEl.className = "readonly-label";
            readOnlyEl.innerHTML = value;
            controlItem.appendChild(readOnlyEl);

            return controlItem;
        }


        // Controles de entrenamiento - Primera fila
        let trainingRow1 = document.createElement("div");
        trainingRow1.className = "control-group";

        trainingRow1.appendChild(createSelectControl(
            "Capas Ocultas", [1, 2], selections.training.hiddenLayers,
            (value) => {
                selections.training.hiddenLayers = value;
                // Check if the second neurons per layer control should be enabled/disabled
                const neuronsPerLayer2Select = document.getElementById('neuronsPerLayer2-select');

                // If disabled, set its value to 0
                if (value === 1) {
                   neuronsPerLayer2Select.disabled = true
                   neuronsPerLayer2Select.value = 0
                   selections.training.neuronsPerLayer2 = 0;
                }else{
                    neuronsPerLayer2Select.disabled = false
                    neuronsPerLayer2Select.value = 5;
                    selections.training.neuronsPerLayer2 = 5;
                }

                window.showTrainingConfig();
            },
            'hiddenLayers-select' // Add ID
        ));

        trainingRow1.appendChild(createSelectControl(
            "Neuronas por Capa", [5, 6, 7, 8, 9, 10], selections.training.neuronsPerLayer,
            (value) => {
                selections.training.neuronsPerLayer = value;
                window.showTrainingConfig();
            },
            'neuronsPerLayer-select' // Add ID
        ));

         trainingRow1.appendChild(createSelectControl(
              "Neuronas por Capa 2", [5, 6, 7, 8, 9, 10], selections.training.neuronsPerLayer2,
              (value) => {
                  selections.training.neuronsPerLayer2 = value;
                  window.showTrainingConfig();
              },
              'neuronsPerLayer2-select', // Add ID
              selections.training.hiddenLayers === 1 // Disable based on initial hiddenLayers
          ));


        // Initial state check for Neuronas por Capa 2
        const hiddenLayersSelect = document.getElementById('hiddenLayers-select');
        const neuronsPerLayerSelect = document.getElementById('neuronsPerLayer-select');

        const neuronsPerLayer2Select = document.getElementById('neuronsPerLayer2-select');
        if (hiddenLayersSelect && neuronsPerLayerSelect) {
             neuronsPerLayer2Select.disabled = (parseInt(hiddenLayersSelect.value) === 1);
             if (neuronsPerLayer2Select.disabled) { // Ensure value is 0 if disabled initially
                neuronsPerLayer2Select.value = 0;
                selections.training.neuronsPerLayer2 = 0;
            }
        }

        trainingSection.appendChild(trainingRow1);

        // Controles de entrenamiento - Segunda fila
        let trainingRow2 = document.createElement("div");
        trainingRow2.className = "control-group";

        trainingRow2.appendChild(createNumberInput(
            "Término Momento", selections.training.momentum,
            (value) => {
                selections.training.momentum = value;
                window.showTrainingConfig();
            }, 0.0, 1.0, 0.1
        ));

        trainingRow2.appendChild(createNumberInput(
            "Coeficiente de Aprendizaje", selections.training.learningRate,
            (value) => {
                selections.training.learningRate = value;
                window.showTrainingConfig();
            },
             0.0, 1.0, 0.1
        ));

        trainingRow2.appendChild(createNumberInput(
            "Número de Épocas", selections.training.epochs,
            (value) => {
                selections.training.epochs = parseInt(value) || 1;
                window.showTrainingConfig();
            }, 1
        ));

        trainingSection.appendChild(trainingRow2);

        // Control Crear boton
        let trainingRow3 = document.createElement("div");
        trainingRow3.className = "control-group";

        // Botón para crear el perceptron
        let createButton = document.createElement("button");
        createButton.className = "clear-button";
        createButton.style.backgroundColor = "#2ABD36"
        createButton.innerHTML = "Crear";
        createButton.addEventListener("click", () => {
            const config = {
                hidden_layers_count: selections.training.hiddenLayers,
                neurons_layer1: selections.training.neuronsPerLayer,
                neurons_layer2: selections.training.neuronsPerLayer2,
                learning_rate: selections.training.learningRate,
                momentum: selections.training.momentum,
                epochs: selections.training.epochs
            };
            model.send({ event: 'create_mlp', config: config });
            window.updateResults("Creando MLP...", 'info', trainingDisplayInfo);
        });

        let controls = document.createElement("div");
        controls.style.textAlign = "center";
        controls.appendChild(createButton);
        trainingRow3.appendChild(controls);

        trainingSection.appendChild(trainingRow3);

       // Add the trainingDisplayInfo element to the training section
        trainingSection.appendChild(trainingDisplayInfo);

       // ================= SECCIÓN 2: DATASET =======================================
        let dataSetSection = document.createElement("div");
        dataSetSection.className = "section";

        let dataSetdataSetTitle = document.createElement("h2");
        dataSetdataSetTitle.className = "section-title";
        dataSetdataSetTitle.innerHTML = "Dataset de Entrenamiento";
        dataSetSection.appendChild(dataSetdataSetTitle);

        // Controles de dataset
        let dataSetRow = document.createElement("div");
        dataSetRow.className = "control-group";

        dataSetRow.appendChild(createSelectControl(
            "Datasets", [100, 500, 1000], selections.training.dataset,
            (value) => {
                selections.training.dataset = value;
                window.showDataSetConfig();
            }
        ));

        dataSetRow.appendChild(createSelectControl(
            "Validación %", [10, 20, 30], selections.training.validationPercent,
            (value) => {
                selections.training.validationPercent = value;
                window.showDataSetConfig();
            }
        ));

        dataSetSection.appendChild(dataSetRow);

        // Crear etiqueta de resultados para dataset ANTES de crear el botón
        let dataSetDisplayInfo = createDisplayInfo("Seleccione un dataset...", "info");
        dataSetSection.appendChild(dataSetDisplayInfo);

        // Control Crear boton
        let dataSetRow2 = document.createElement("div");
        dataSetRow2.className = "control-group";

        // Botón para entrenar
        let trainingButton = document.createElement("button");
        trainingButton.className = "clear-button";
        trainingButton.style.backgroundColor = "#2A65BD"
        trainingButton.innerHTML = "Entrenar";
        trainingButton.addEventListener("click", () => {
            console.log("Training button clicked");
            const cantidad = selections.training.dataset;
            const validationPercent = selections.training.validationPercent;
            const config = {
                cantidad: cantidad,
                validationPercent: validationPercent
            };
            console.log("Sending train_mlp event with config:", config);
            model.send({ event: 'train_mlp', config: config });
            window.updateResults(`Entrenando MLP con ${cantidad} ejemplos (${validationPercent}% validación)...`, 'info', dataSetDisplayInfo);
        });

        let dataSetControls = document.createElement("div");
        dataSetControls.style.textAlign = "center";
        dataSetControls.appendChild(trainingButton);
        dataSetRow2.appendChild(dataSetControls);

        dataSetSection.appendChild(dataSetRow2);


        // === SECCIÓN 3: MATRIZ 10x10 ===


        let matrixSection = document.createElement("div");
        matrixSection.className = "section";

        let matrixTitle = document.createElement("h2");
        matrixTitle.className = "section-title";
        matrixTitle.innerHTML = "Matriz de Entrada 10x10";
        matrixSection.appendChild(matrixTitle);

        // Selector de letras
        let letterSelector = document.createElement("div");
        letterSelector.className = "letter-selector";

        let letterLabel = document.createElement("div");
        letterLabel.style.cssText = "margin-bottom: 10px; font-weight: bold; color: #333;";
        letterLabel.innerHTML = "Elegir Letra:";
        letterSelector.appendChild(letterLabel);

        let selectedLetter = null;

        // Definir patrones de letras (10x10)
        const letterPatterns = {
            'B': [
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]
            ],
            'D': [
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,1,1,1,1,1,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,0,0,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0]
            ],
            'F': [
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0],
                [0,0,0,0,1,0,0,1,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]
            ]
        };

        // Crear botones para cada letra
        ['B', 'D', 'F'].forEach(letter => {
            let button = document.createElement("button");
            button.className = "letter-button";
            button.innerHTML = letter;
            button.addEventListener("click", () => {
                // Desseleccionar otros botones
                letterSelector.querySelectorAll('.letter-button').forEach(btn => {
                    btn.classList.remove('selected');
                });
                // Seleccionar este botón
                button.classList.add('selected');
                selectedLetter = letter;

                // Dibujar la letra en la matriz
                drawLetterInMatrix(letter);
            });
            letterSelector.appendChild(button);
        });

        matrixSection.appendChild(letterSelector);

        let distortionRow = document.createElement("div");
        distortionRow.className = "control-group";
        distortionRow.style.alignItems = "flex-end";

        distortionRow.appendChild(createNumberInput(
            "Distorsión %", selections.training.distortionPercent,
            (value) => {
                selections.training.distortionPercent = value;
                window.showTrainingConfig();
            }, 0.0, 0.3, 0.01
        ));

        // Botón para aplicar distorsión (el evento se agregará después de crear cells)
        let applyDistortionButton = document.createElement("button");
        applyDistortionButton.className = "clear-button";
        applyDistortionButton.style.backgroundColor = "#9C27B0";
        applyDistortionButton.style.marginTop = "0";
        applyDistortionButton.innerHTML = "Aplicar Distorsión";

        let distortionButtonContainer = document.createElement("div");
        distortionButtonContainer.className = "control-item";
        distortionButtonContainer.appendChild(applyDistortionButton);
        distortionRow.appendChild(distortionButtonContainer);

        matrixSection.appendChild(distortionRow);

        let separator = document.createElement("hr");
        matrixSection.appendChild(separator);


        // Crear la matriz
        let matrixContainer = document.createElement("div");
        matrixContainer.className = "matrix-container";

        // Crear etiqueta para mostrar letra seleccionada
        let selectedLetterLabel = createDisplayInfo("Letra seleccionada: Ninguna", "info");
        matrixSection.appendChild(selectedLetterLabel);

        let cells = [];

        for (let i = 0; i < 10; i++) {
            cells[i] = [];
            for (let j = 0; j < 10; j++) {
                let cell = document.createElement("div");
                cell.className = "matrix-cell";

                // cell.addEventListener("click", () => {
                //     if (cell.style.backgroundColor === "white" || cell.style.backgroundColor === "") {
                //         cell.style.backgroundColor = "#2196F3";
                //     } else {
                //         cell.style.backgroundColor = "white";
                //     }

                //     // Actualizar el estado en el modelo
                //     let matrix = model.get("matrix");
                //     matrix[i][j] = cell.style.backgroundColor === "white" || cell.style.backgroundColor === "" ? 0 : 1;
                //     model.set("matrix", matrix);
                //     model.save_changes();

                //     // Actualizar la información de la matriz
                //     window.showMatrixInfo();
                // });

                cells[i][j] = cell;
                matrixContainer.appendChild(cell);
            }
        }

        // Función para actualizar la matriz desde el modelo
        let updateMatrix = () => {
            let matrix = model.get("matrix");
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    if (matrix[i][j] === 1) {
                        cells[i][j].style.backgroundColor = "#2196F3";
                    } else {
                        cells[i][j].style.backgroundColor = "white";
                    }
                }
            }
        };

        model.on("change:matrix", updateMatrix);
        updateMatrix();

        // Función para dibujar una letra en la matriz
        function drawLetterInMatrix(letter) {
            const pattern = letterPatterns[letter];
            if (!pattern) return;

            // Crear nueva matriz con el patrón de la letra
            let matrix = Array(10).fill().map(() => Array(10).fill(0));
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    matrix[i][j] = pattern[i][j];
                }
            }

            // Actualizar el modelo
            model.set("matrix", matrix);
            model.save_changes();

            // Mostrar mensaje
            window.updateResults(`Letra '${letter}' dibujada en la matriz`, 'success', selectedLetterLabel);
        }

        // Agregar evento al botón de distorsión (ahora que cells y updateMatrix están definidos)
        applyDistortionButton.addEventListener("click", () => {
            let matrix = model.get("matrix");
            let distortionPercent = selections.training.distortionPercent;

            // Contar celdas activas
            let activeCells = 0;
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    if (matrix[i][j] === 1) activeCells++;
                }
            }

            if (activeCells === 0) {
                window.updateResults("⚠️ No hay celdas activas para distorsionar", 'warning', selectedLetterLabel);
                return;
            }

            // Calcular número de celdas a distorsionar
            let cellsToDistort = Math.floor(100 * distortionPercent);

            if (cellsToDistort === 0) {
                window.updateResults("⚠️ Porcentaje de distorsión muy bajo (0 celdas)", 'warning', selectedLetterLabel);
                return;
            }

            // Aplicar distorsión aleatoria
            let distortedCount = 0;
            let attempts = 0;
            let maxAttempts = cellsToDistort * 10; // Evitar bucle infinito

            while (distortedCount < cellsToDistort && attempts < maxAttempts) {
                let i = Math.floor(Math.random() * 10);
                let j = Math.floor(Math.random() * 10);

                // Invertir el valor de la celda
                matrix[i][j] = matrix[i][j] === 1 ? 0 : 1;
                distortedCount++;
                attempts++;
            }

            // Actualizar el modelo
            model.set("matrix", matrix);
            model.save_changes();

            // Actualizar la visualización de las celdas
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    if (matrix[i][j] === 1) {
                        cells[i][j].style.backgroundColor = "#2196F3";
                    } else {
                        cells[i][j].style.backgroundColor = "white";
                    }
                }
            }

            // Mostrar mensaje
            window.updateResults(`🎲 Distorsión aplicada: ${cellsToDistort} celdas modificadas (${(distortionPercent * 100).toFixed(1)}%)`, 'info', selectedLetterLabel);
        });

        // Botón para limpiar matriz
        let clearButton = document.createElement("button");
        clearButton.className = "clear-button";
        clearButton.innerHTML = "Limpiar Matriz";
        clearButton.addEventListener("click", () => {
            let matrix = Array(10).fill().map(() => Array(10).fill(0));
            model.set("matrix", matrix);
            model.save_changes();

            // Deseleccionar letra
            selectedLetter = null;
            letterSelector.querySelectorAll('.letter-button').forEach(btn => {
                btn.classList.remove('selected');
            });

            window.updateResults("🧹 Matriz limpiada - 0/100 celdas activas", 'success', selectedLetterLabel);
        });

        let matrixControls = document.createElement("div");
        matrixControls.style.textAlign = "center";
        matrixControls.appendChild(clearButton);

        // boton para predecir
        let predictButton = document.createElement("button");
        predictButton.className = "clear-button";
        predictButton.style.backgroundColor = "#FF9800"
        predictButton.innerHTML = "Predecir";
        predictButton.addEventListener("click", () => {
            let matrix = model.get("matrix");
            
            // Verificar que la matriz no esté vacía
            let activeCount = 0;
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    if (matrix[i][j] === 1) activeCount++;
                }
            }
            
            if (activeCount === 0) {
                window.updateResults("⚠️ La matriz está vacía. Dibuje una letra primero.", 'warning', selectedLetterLabel);
                return;
            }
            
            // Enviar matriz al backend para clasificación
            model.send({ event: 'predict', matrix: matrix });
            window.updateResults("🔮 Clasificando patrón...", 'info', selectedLetterLabel);
        });
        matrixControls.appendChild(predictButton);


        matrixSection.appendChild(matrixContainer);
        matrixSection.appendChild(matrixControls);

        // Agregar todas las secciones al contenedor principal
        mainContainer.appendChild(trainingSection);
        mainContainer.appendChild(dataSetSection);
        mainContainer.appendChild(matrixSection);

        el.appendChild(mainContainer);

        // Mostrar configuración inicial
        setTimeout(() => {
            window.showTrainingConfig();
            window.showDataSetConfig();
            window.showMatrixInfo();
        }, 100);

        // Manejar mensajes desde Python
        model.on('msg:custom', (msg) => {
            console.log("Received message from Python:", msg);
            if (msg.event === 'display_message') {
                // Determine which display info to use based on 'target'
                let targetDisplay;
                if (msg.target === 'dataset') {
                    console.log("Target is dataset");
                    targetDisplay = dataSetDisplayInfo;
                } else if (msg.target === 'matrix') {
                    console.log("Target is matrix");
                    targetDisplay = selectedLetterLabel;
                } else {
                    console.log("Target is training (default)");
                    targetDisplay = trainingDisplayInfo; // Default to training
                }
                console.log("Updating results with:", msg.text, msg.type, targetDisplay);
                window.updateResults(msg.text, msg.type, targetDisplay);
            }
        });

        // Función para obtener todas las selecciones
        window.getSelections = () => selections;

        // Función para actualizar los resultados
        window.updateResults = (message, type = 'info', resultLabel = trainingDisplayInfo) => {
            let colors = {
                'info': { bg: '#e3f2fd', border: '#2196F3', text: '#1976D2' },
                'success': { bg: '#e8f5e8', border: '#4caf50', text: '#2e7d32' },
                'error': { bg: '#ffebee', border: '#f44336', text: '#c62828' },
                'warning': { bg: '#fff3e0', border: '#ff9800', text: '#ef6c00' }
            };

            let color = colors[type] || colors['info'];
            resultLabel.style.backgroundColor = color.bg;
            resultLabel.style.borderColor = color.border;
            resultLabel.style.color = color.text;
            resultLabel.innerHTML = message;
        };

        // Función para mostrar el estado de la matriz
        window.showMatrixInfo = () => {
            let matrix = model.get("matrix");
            let activeCount = 0;
            for (let i = 0; i < 10; i++) {
                for (let j = 0; j < 10; j++) {
                    if (matrix[i][j] === 1) activeCount++;
                }
            }
            // Changed result label to selectedLetterLabel for matrix info
            window.updateResults(`Matriz: ${activeCount}/100 celdas activas`, 'info', selectedLetterLabel);
        };

        // Función para mostrar configuración de entrenamiento
        window.showTrainingConfig = () => {
            let config = selections.training;
            let message = `Entrenamiento: ${config.hiddenLayers} capas, ${config.neuronsPerLayer} neuronas/capa 1, ${config.neuronsPerLayer2 > 0 ? config.neuronsPerLayer2 + ' neuronas/capa 2, ' : ''}Coef. Aprendizaje: ${config.learningRate}, Momento: ${config.momentum}, Épocas: ${config.epochs}`;
            window.updateResults(message, 'info', trainingDisplayInfo);
        };

        // Función para mostrar dataset de entrenamiento
        window.showDataSetConfig = () => {
            let config = selections.training;
            let message = `Dataset: ${config.dataset} ejemplos, Validación: ${config.validationPercent}%`;
            window.updateResults(message, 'warning', dataSetDisplayInfo);
        };

        // Función para dibujar una letra específica (accesible desde Python)
        window.drawLetter = (letter) => {
            if (['B', 'D', 'F'].includes(letter.toUpperCase())) {
                drawLetterInMatrix(letter.toUpperCase());
                // Seleccionar el botón correspondiente
                letterSelector.querySelectorAll('.letter-button').forEach(btn => {
                    btn.classList.remove('selected');
                    if (btn.innerHTML === letter.toUpperCase()) {
                        btn.classList.add('selected');
                    }
                });
                selectedLetter = letter.toUpperCase();
                return true;
            }
            return false;
        };

        // Función para obtener la letra seleccionada
        window.getSelectedLetter = () => selectedLetter;
    }
    export default { render };
    """

    # Propiedades del modelo
    matrix = traitlets.List(
        [list(range(10)) for _ in range(10)]
    ).tag(sync=True)

    # Parámetros de entrenamiento
    hidden_layers = traitlets.Int(1).tag(sync=True)
    neurons_per_layer = traitlets.Int(5).tag(sync=True)
    neurons_per_layer2 = traitlets.Int(5).tag(sync=True)
    dataset_size = traitlets.Int(100).tag(sync=True)
    training_percent = traitlets.Int(50).tag(sync=True)
    validation_percent = traitlets.Int(10).tag(sync=True)
    momentum = traitlets.Float(0.5).tag(sync=True)
    epochs = traitlets.Int(100).tag(sync=True)

    # Parámetros de inferencia
    inference_hidden_layers = traitlets.Int(1).tag(sync=True)
    inference_neurons_per_layer = traitlets.Int(1).tag(sync=True)
    selected_letter = traitlets.Unicode('b').tag(sync=True)
    dispersion_percent = traitlets.Int(15).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inicializar la matriz con ceros
        self.matrix = [[0 for _ in range(10)] for _ in range(10)]
        self._created_mlp = None # To store the MLP instance
        self.generador = GeneradorDataset() # Initialize GeneradorDataset
        self.clasificador = None # Clasificador de letras

        # Register the message handler
        self.on_msg(self._handle_frontend_message)

    def _handle_frontend_message(self, widget, content, buffers):
        print(f"=== MESSAGE HANDLER CALLED ===")
        print(f"Received message from frontend: {content}")
        print(f"Event type: {content.get('event')}")

        if content.get('event') == 'create_mlp':
            config = content.get('config')
            print(f"Handling create_mlp with config: {config}")
            self._create_mlp_from_config(config)
        elif content.get('event') == 'train_mlp': # Changed event name
            config = content.get('config')
            print(f"Handling train_mlp with config: {config}")
            print(f"Cantidad from config: {config.get('cantidad') if config else 'Config is None'}")
            print(f"ValidationPercent from config: {config.get('validationPercent') if config else 'Config is None'}")
            if config:
                cantidad = config.get('cantidad')
                validation_percent = config.get('validationPercent', 10)  # Default 10% si no se proporciona
                self._train_mlp(cantidad, validation_percent)
            else:
                print("ERROR: Config is None!")
                self.show_error_message("Error: configuración no recibida", target='dataset')
        elif content.get('event') == 'predict':
            matrix = content.get('matrix')
            print(f"Handling predict with matrix")
            self._predict_pattern(matrix)
        else:
            print(f"Unknown event: {content.get('event')}")

    def _create_mlp_from_config(self, config):
        try:
            hidden_layers_count = config['hidden_layers_count']
            neurons_layer1 = config['neurons_layer1']
            neurons_layer2 = config['neurons_layer2'] # This value will be 0 if hidden_layers_count is 1
            learning_rate = config['learning_rate']
            momentum = config['momentum']
            epochs = config['epochs']

            # Determine capas_ocultas based on hidden_layers_count
            if hidden_layers_count == 1:
                capas_ocultas = [neurons_layer1]
            elif hidden_layers_count == 2:
                capas_ocultas = [neurons_layer1, neurons_layer2]
            else:
                raise ValueError("Número de capas ocultas debe ser 1 o 2.")

            self._created_mlp = mlp.MLP(
                capas_ocultas=capas_ocultas,
                learning_rate=learning_rate,
                momentum=momentum,
                epochs=epochs
            )
            mensaje = f"MLP creado exitosamente!\nArquitectura: {self._created_mlp.arquitectura} neuronas\nÉpocas: {epochs}, Learning Rate: {learning_rate}, Momentum: {momentum}"
            self.show_success_message(mensaje, target='training')
        except ValueError as e:
            self.show_error_message(f"Error al crear MLP: {e}", target='training')
        except Exception as e:
            self.show_error_message(f"Error inesperado al crear MLP: {e}", target='training')

    def _generate_dataset(self, cantidad):
        try:
            print(f"Starting dataset generation for {cantidad} examples...")
            # Ensure original datasets are generated first

            self.generador.generar_dataset_equilibrado(
                cant=cantidad,
                min_distorsion=1,
                max_distorsion=30,
                metodo_v2=False
            )

            print("Generated balanced dataset")

            message = f"Dataset de {cantidad} ejemplos generado exitosamente."

            print(f"Sending success message: {message}")

            self.show_success_message(message, target='dataset')

        except Exception as e:
            print(f"Error generating dataset: {e}")
            self.show_error_message(f"Error al generar el dataset: {e}", target='dataset')

    def _train_mlp(self, cantidad, validation_percent=10):
        print(f"=== _train_mlp CALLED with cantidad={cantidad}, validation_percent={validation_percent}% ===")
        print(f"MLP instance: {self._created_mlp}")

        if self._created_mlp is None:
            print("ERROR: MLP not created yet!")
            self.show_error_message("Primero debe crear un MLP. Haga clic en 'Crear' en la sección de Configuración de Entrenamiento.", target='dataset')
            return

        try:
            print(f"Starting training process...")
            self.show_success_message(f"Generando dataset de {cantidad} ejemplos antes de entrenar...", target='dataset')
            # Ensure dataset is generated before training
            self._generate_dataset(cantidad)

            print(f"Dataset generated, now loading and splitting (validation: {validation_percent}%)...")
            self.show_success_message(f"Cargando dataset y separando {validation_percent}% para validación...", target='dataset')

            # Load dataset
            X, y = leer_dataset(cantidad, 'distorsionadas')
            print(f"Data loaded, X shape: {X.shape}, y shape: {y.shape}")

            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=validation_percent / 100.0,
                random_state=42
            )

            training_size = len(X_train)
            validation_size = len(X_val)
            print(f"Training set: {training_size} ejemplos, Validation set: {validation_size} ejemplos")

            # Train the MLP
            print(f"Starting MLP training...")
            self.show_success_message(f"Entrenando MLP con {training_size} ejemplos ({validation_size} para validación)...", target='dataset')

            historial = self._created_mlp.entrenar(
                X_train, y_train,          # Datos de entrenamiento
                X_val=X_val,               # Datos de validación
                y_val=y_val,
                verbose=True
            )

            print(f"MLP training completed!")

            # Graficar MSE si el historial está disponible
            if historial:
                try:
                    graficar_mse_entrenamiento_validacion(
                        historial,
                        titulo=f"MSE de Entrenamiento vs Validación - {self._created_mlp.arquitectura}"
                    )
                except Exception as e_viz:
                    print(f"Warning: Could not generate visualization: {e_viz}")

            self.show_success_message(f"Entrenamiento completado exitosamente con {training_size} ejemplos de entrenamiento.", target='dataset')
            
            # Crear clasificador con el MLP entrenado
            from clasificador import ClasificadorLetras
            self.clasificador = ClasificadorLetras(self._created_mlp)
            print("✅ Clasificador creado y listo para usar")

        except Exception as e:
            print(f"ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message(f"Error durante el entrenamiento del MLP: {e}", target='dataset')


    def get_matrix(self):
        """Obtener el estado actual de la matriz"""
        return self.matrix

    def clear_matrix(self):
        """Limpiar la matriz (poner todas las celdas en blanco)"""
        self.matrix = [[0 for _ in range(10)] for _ in range(10)]

    def get_training_config(self):
        """Obtener configuración de entrenamiento"""
        return {
            'hidden_layers': self.hidden_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'neurons_per_layer2': self.neurons_per_layer2,
            'dataset_size': self.dataset_size,
            'training_percent': self.training_percent,
            'validation_percent': self.validation_percent,
            'momentum': self.momentum,
            'epochs': self.epochs,
            'learning_rate': 0.5  # Coeficiente fijo
        }

    def get_inference_config(self):
        """Obtener configuración de inferencia"""
        return {
            'hidden_layers': self.inference_hidden_layers,
            'neurons_per_layer': self.inference_neurons_per_layer,
            'selected_letter': self.selected_letter,
            'dispersion_percent': self.dispersion_percent
        }

    def print_matrix(self):
        """Imprimir la matriz en formato legible"""
        print("Estado actual de la matriz:")
        for i, row in enumerate(self.matrix):
            print(f"Fila {i}: {row}")

    def print_all_config(self):
        """Imprimir toda la configuración"""
        print("=== CONFIGURACIÓN DE ENTRENAMIENTO ===")
        config = self.get_training_config()
        for key, value in config.items():
            print(f"{key}: {value}")

        print("\n=== CONFIGURACIÓN DE INFERENCIA ===")
        config = self.get_inference_config()
        for key, value in config.items():
            print(f"{key}: {value}")

    def show_training_results(self, accuracy=None, loss=None, epoch=None):
        """Mostrar resultados del entrenamiento"""
        if accuracy is not None and loss is not None and epoch is not None:
            message = f"🎯 Época {epoch}: Precisión {accuracy:.2%}, Pérdida {loss:.4f}"
            self.send({'event': 'display_message', 'text': message, 'type': 'info', 'target': 'training'})

    def show_prediction_results(self, predicted_letter=None, confidence=None):
        """Mostrar resultados de predicción"""
        if predicted_letter and confidence is not None:
            message = f"🔮 Predicción: Letra '{predicted_letter}' (Confianza: {confidence:.2%})"
            self.send({'event': 'display_message', 'text': message, 'type': 'info', 'target': 'matrix'})

    def show_error_message(self, error_text, target='training'):
        """Mostrar mensaje de error"""
        print(f"=== SENDING ERROR MESSAGE ===")
        print(f"Text: {error_text}, Target: {target}")
        self.send({'event': 'display_message', 'text': f"❌ Error: {error_text}", 'type': 'error', 'target': target})
        print(f"Message sent!")

    def show_success_message(self, success_text, target='training'):
        """Mostrar mensaje de éxito"""
        print(f"=== SENDING SUCCESS MESSAGE ===")
        print(f"Text: {success_text}, Target: {target}")
        self.send({'event': 'display_message', 'text': f"✅ {success_text}", 'type': 'success', 'target': target})
        print(f"Message sent!")

    def update_training_display(self):
        """Actualizar la visualización de configuración de entrenamiento"""
        config = self.get_training_config();
        #message = `🧠 Entrenamiento: ${config['hidden_layers']} capas, ${config['neurons_per_layer']} neuronas/capa 1, ${config['neurons_per_layer2']} neuronas/capa 2, Dataset: ${config['dataset_size']}, Entrena: ${config['training_percent']}%, Valida: ${config['validation_percent']}%, Momento: ${config['momentum']}, Épocas: ${config['epochs']}`;
        #self.send({'event': 'display_message', 'text': "Configuración de Entrenamiento: " + message, 'type': 'info', 'target': 'training'})

    def update_inference_display(self):
        """Actualizar la visualización de configuración de inferencia"""
        config = self.get_inference_config()
        message = f"🔍 Inferencia: {config['hidden_layers']} capas, {config['neurons_per_layer']} neuronas/capa, Letra: '{config['selected_letter']}', Dispersión: {config['dispersion_percent']}%"
        self.send({'event': 'display_message', 'text': message, 'type': 'info', 'target': 'matrix'})

    def show_current_config(self):
        """Mostrar toda la configuración actual"""
        self.send({'event': 'display_message', 'text': "=== CONFIGURACIÓN ACTUAL ===", 'type': 'info', 'target': 'training'})
        self.update_training_display()
        self.update_inference_display()
        matrix_count = sum(sum(row) for row in self.matrix)
        self.send({'event': 'display_message', 'text': f"📊 Matriz: {matrix_count}/100 celdas activas", 'type': 'info', 'target': 'matrix'})

    def draw_letter(self, letter):
        """Dibujar una letra específica en la matriz"""
        if letter.upper() in ['B', 'D', 'F']:
            self.send({'event': 'display_message', 'text': f"✏️ Dibujando letra '{letter.upper()}' en la matriz...", 'type': 'info', 'target': 'matrix'})
            # La función JavaScript se encargará del dibujo real
            return True
        else:
            self.send({'event': 'display_message', 'text': f"❌ Error: Letra '{letter}' no disponible. Use B, D o F.", 'type': 'error', 'target': 'matrix'})
            return False

    def get_available_letters(self):
        """Obtener lista de letras disponibles"""
        return ['B', 'D', 'F']

    def clear_letter_selection(self):
        """Limpiar la selección de letra y la matriz"""
        self.clear_matrix()
        self.send({'event': 'display_message', 'text': "🧹 Matriz y selección de letra limpiadas", 'type': 'success', 'target': 'matrix'})
    
    def _predict_pattern(self, matrix):
        """Clasifica el patrón de la matriz y muestra los resultados"""
        try:
            if self.clasificador is None:
                self.show_error_message("Primero debe entrenar el modelo. Haga clic en 'Entrenar' en la sección Dataset.", target='matrix')
                return
            
            # Convertir matriz a numpy array y aplanar
            import numpy as np
            patron = np.array(matrix).flatten()
            
            print(f"Patrón a clasificar: {patron.shape}")
            print(f"Celdas activas: {np.sum(patron)}")
            
            # Clasificar el patrón
            resultado = self.clasificador.clasificar_patron(patron)
            
            print(f"Resultado de clasificación: {resultado}")
            
            # Formatear mensaje con probabilidades
            prob_b = resultado['probabilidades'][0] * 100
            prob_d = resultado['probabilidades'][1] * 100
            prob_f = resultado['probabilidades'][2] * 100
            confianza = resultado['confianza'] * 100
            
            mensaje = f"""🔮 RESULTADO DE CLASIFICACIÓN
            
              Predicción: {resultado['letra']} ({confianza:.1f}% confianza)
                          
              Probabilidades:
              📊 B: {prob_b:.1f}%
              📊 D: {prob_d:.1f}%
              📊 F: {prob_f:.1f}%"""
            
            self.send({'event': 'display_message', 'text': mensaje, 'type': 'success', 'target': 'matrix'})
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message(f"Error al clasificar el patrón: {e}", target='matrix')

# Crear y mostrar el widget
def create_neural_network_interface():
    """Función para crear una nueva interfaz de red neuronal"""
    return NeuralNetworkInterface()




print("Interfaz de Red Neuronal creada.")
print("Para mostrar la interfaz en Jupyter/Colab, ejecuta: nn_widget")
print("Métodos disponibles:")
print("- nn_widget.get_matrix(): Obtener estado de la matriz")
print("- nn_widget.clear_matrix(): Limpiar la matriz")
print("- nn_widget.get_training_config(): Obtener configuración de entrenamiento")
print("- nn_widget.get_inference_config(): Obtener configuración de inferencia")
print("- nn_widget.print_matrix(): Imprimir matriz en consola")
print("- nn_widget.print_all_config(): Imprimir toda la configuración")
