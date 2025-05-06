 // Global variables
    let wasmModule = null;
    let hiddenLayers = [];
    let networkBuilt = false;

    // Activation function options
    const activationOptions = [
    { value: 0, label: "ReLU" },
    { value: 1, label: "Sigmoid" },
    { value: 2, label: "Tanh" },
    { value: 3, label: "Leaky ReLU" },
    { value: 4, label: "Swish" },
    { value: 5, label: "Linear" }
    ];

    // Initialize the page
    document.addEventListener('DOMContentLoaded', function() {
    // Set up tab navigation
    setupTabs();

    // Load the WASM module
    createNeuralNetworkModule().then(module => {
    wasmModule = module;
    const status = document.getElementById('status');
    status.className = "status success";
    status.textContent = "WASM module loaded successfully!";

    // Add first hidden layer by default
    addHiddenLayer();

    // Add event listeners
    document.getElementById('addLayerBtn').addEventListener('click', addHiddenLayer);
    document.getElementById('buildNetworkBtn').addEventListener('click', buildNetwork);
    document.getElementById('trainNetworkBtn').addEventListener('click', trainNetwork);

    // Set up example buttons
    setupExampleButtons();

}).catch(error => {
    console.error("Failed to load WASM module:", error);
    const status = document.getElementById('status');
    status.className = "status error";
    status.textContent = "Failed to load WASM module: " + error;
});
});

    // Set up example buttons
    function setupExampleButtons() {
    const exampleButtons = document.querySelectorAll('.example-btn');
    exampleButtons.forEach(button => {
    button.addEventListener('click', function() {
    const layersData = JSON.parse(this.getAttribute('data-layers').replace(/'/g, '"'));
    clearLayers();

    // Add layers from the example
    layersData.forEach(layer => {
    addHiddenLayer(layer.neurons, layer.activation);
});
});
});
}

    // Clear all layers
    function clearLayers() {
    hiddenLayers = [];
    document.getElementById('layersContainer').innerHTML = '';
}

    // Set up tab navigation
    function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
    tab.addEventListener('click', () => {
    // Remove active class from all tabs and contents
    tabs.forEach(t => t.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));

    // Add active class to clicked tab and corresponding content
    tab.classList.add('active');
    const tabName = tab.getAttribute('data-tab');
    document.getElementById(tabName + 'Tab').classList.add('active');
});
});
}

    // Function to add a new hidden layer
    function addHiddenLayer(neurons = 8, activation = 3) {
    const layerIndex = hiddenLayers.length;

    // Create default configuration
    hiddenLayers.push({
    neurons: neurons,
    activation: activation
});

    // Create the UI element
    const layerContainer = document.getElementById('layersContainer');
    const layerDiv = document.createElement('div');
    layerDiv.className = 'layer-container';
    layerDiv.dataset.index = layerIndex;

    const layerHTML = `
                <div class="layer-header">
                    <h3>Hidden Layer ${layerIndex + 1}</h3>
                    <button class="remove-btn" data-index="${layerIndex}" ${layerIndex === 0 ? 'disabled' : ''}>Remove</button>
                </div>
                <div class="layer-settings">
                    <div>
                        <label for="neurons-${layerIndex}">Number of Neurons:</label>
                        <input type="number" id="neurons-${layerIndex}" min="1" max="100" value="${neurons}" data-index="${layerIndex}">
                    </div>
                    <div>
                        <label for="activation-${layerIndex}">Activation Function:</label>
                        <select id="activation-${layerIndex}" data-index="${layerIndex}">
                            ${activationOptions.map(opt =>
    `<option value="${opt.value}" ${opt.value === activation ? 'selected' : ''}>${opt.label}</option>`
    ).join('')}
                        </select>
                    </div>
                </div>
            `;

    layerDiv.innerHTML = layerHTML;
    layerContainer.appendChild(layerDiv);

    // Add event listeners
    const neuronInput = document.getElementById(`neurons-${layerIndex}`);
    neuronInput.addEventListener('change', function() {
    hiddenLayers[this.dataset.index].neurons = parseInt(this.value);
});

    const activationSelect = document.getElementById(`activation-${layerIndex}`);
    activationSelect.addEventListener('change', function() {
    hiddenLayers[this.dataset.index].activation = parseInt(this.value);
});

    const removeBtn = layerDiv.querySelector('.remove-btn');
    removeBtn.addEventListener('click', function() {
    removeHiddenLayer(parseInt(this.dataset.index));
});
}

    // Function to remove a hidden layer
    function removeHiddenLayer(index) {
    // Remove from the array
    hiddenLayers.splice(index, 1);

    // Update the UI
    const layerContainer = document.getElementById('layersContainer');
    layerContainer.innerHTML = '';

    // Recreate all layers with updated indices
    hiddenLayers.forEach((layer, i) => {
    const layerDiv = document.createElement('div');
    layerDiv.className = 'layer-container';
    layerDiv.dataset.index = i;

    const layerHTML = `
                    <div class="layer-header">
                        <h3>Hidden Layer ${i + 1}</h3>
                        <button class="remove-btn" data-index="${i}" ${i === 0 && hiddenLayers.length === 1 ? 'disabled' : ''}>Remove</button>
                    </div>
                    <div class="layer-settings">
                        <div>
                            <label for="neurons-${i}">Number of Neurons:</label>
                            <input type="number" id="neurons-${i}" min="1" max="100" value="${layer.neurons}" data-index="${i}">
                        </div>
                        <div>
                            <label for="activation-${i}">Activation Function:</label>
                            <select id="activation-${i}" data-index="${i}">
                                ${activationOptions.map(opt =>
    `<option value="${opt.value}" ${opt.value === layer.activation ? 'selected' : ''}>${opt.label}</option>`
    ).join('')}
                            </select>
                        </div>
                    </div>
                `;

    layerDiv.innerHTML = layerHTML;
    layerContainer.appendChild(layerDiv);

    // Add event listeners
    const neuronInput = document.getElementById(`neurons-${i}`);
    neuronInput.addEventListener('change', function() {
    hiddenLayers[this.dataset.index].neurons = parseInt(this.value);
});

    const activationSelect = document.getElementById(`activation-${i}`);
    activationSelect.addEventListener('change', function() {
    hiddenLayers[this.dataset.index].activation = parseInt(this.value);
});

    const removeBtn = layerDiv.querySelector('.remove-btn');
    removeBtn.addEventListener('click', function() {
    removeHiddenLayer(parseInt(this.dataset.index));
});
});
}

    // Function to build the neural network
 // Function to build the neural network
 function buildNetwork() {
     if (!wasmModule) {
         alert("WASM module not loaded yet!");
         return;
     }

     const status = document.getElementById('status');
     status.className = "status loading";
     status.textContent = "Building neural network...";

     try {
         // Standard parameters for 4-bit parity problem
         const inputRows = 16;  // All possible 4-bit patterns
         const inputCols = 4;   // 4 bits per pattern
         const outputRows = 16; // Output for each pattern
         const outputCols = 1;  // Binary classification

         // If we only have one hidden layer, use the simple function
         if (hiddenLayers.length === 1) {
             wasmModule.ccall(
                 'createCustomBlock',
                 null,
                 ['number', 'number', 'number', 'number', 'number', 'number'],
                 [
                     inputRows,
                     inputCols,
                     hiddenLayers[0].neurons,
                     outputRows,
                     outputCols,
                     2  // Cross entropy loss
                 ]
             );
         } else {
             // Multi-layer network
             const numHiddenLayers = hiddenLayers.length;

             // Create arrays for layer sizes and activations
             const sizesArray = new Int32Array(numHiddenLayers);
             const activationsArray = new Int32Array(numHiddenLayers);

             // Fill the arrays
             for (let i = 0; i < numHiddenLayers; i++) {
                 sizesArray[i] = hiddenLayers[i].neurons;
                 activationsArray[i] = hiddenLayers[i].activation;
             }

             // Allocate memory in the WASM heap
             const sizesPtr = wasmModule._malloc(numHiddenLayers * 4); // Int32 = 4 bytes
             const activationsPtr = wasmModule._malloc(numHiddenLayers * 4);

             // Copy JavaScript arrays to WASM memory
             wasmModule.HEAP32.set(sizesArray, sizesPtr / 4);
             wasmModule.HEAP32.set(activationsArray, activationsPtr / 4);

             try {
                 // Call the WASM function
                 wasmModule.ccall(
                     'createMultiLayerNetwork',
                     null,
                     ['number', 'number', 'number', 'number', 'number',
                         'number', 'number', 'number', 'number'],
                     [
                         inputRows,
                         inputCols,
                         numHiddenLayers,
                         sizesPtr,
                         activationsPtr,
                         outputRows,
                         outputCols,
                         2   // Cross entropy loss
                     ]
                 );
             } finally {
                 // Free the allocated memory
                 wasmModule._free(sizesPtr);
                 wasmModule._free(activationsPtr);
             }
         }

         // Get the block size
         const blockSize = wasmModule.ccall('getBlockSize', 'number', [], []);

         // Try to get the loss if available
         let loss = "N/A";
         try {
             loss = wasmModule.ccall('getNetworkLoss', 'number', [], []);
             loss = loss.toFixed(6);
         } catch (e) {
             console.warn("Loss calculation not available:", e);
         }

         // Update status
         status.className = "status success";
         status.textContent = `Network built successfully! Total layers: ${blockSize}`;

         // Update network info
         updateNetworkInfo(blockSize, loss);

         // Render network visualization with safeguards
         setTimeout(() => {
             try {
                 if (window.networkVisualizer) {
                     // Make sure the visualization container exists
                     const container = document.getElementById('network-visualization');
                     if (!container) {
                         // Create the container if it doesn't exist
                         const vizContainer = document.createElement('div');
                         vizContainer.id = 'network-visualization';
                         vizContainer.style.width = '100%';
                         vizContainer.style.height = '400px';
                         vizContainer.style.marginTop = '20px';
                         vizContainer.style.backgroundColor = '#f9f9f9';
                         vizContainer.style.borderRadius = '8px';

                         const networkInfo = document.getElementById('networkInfo');
                         if (networkInfo) {
                             networkInfo.appendChild(vizContainer);
                         }
                     }

                     // Render the visualization
                     window.networkVisualizer.render(hiddenLayers);
                 } else {
                     // Create visualizer if it doesn't exist
                     window.networkVisualizer = new NetworkVisualizer('network-visualization');
                     window.networkVisualizer.render(hiddenLayers);
                 }
             } catch (error) {
                 console.warn('Error rendering visualization:', error);
             }
         }, 100); // Short delay to ensure the DOM is ready

         // Set flag that network is built
         networkBuilt = true;

         // Automatically switch to training tab
         document.querySelector('.tab[data-tab="training"]').click();

     } catch (error) {
         console.error("Error building network:", error);
         status.className = "status error";
         status.textContent = "Error building network: " + error;
         networkBuilt = false;
     }
 }

    // Function to train the network using iterative approach
 // Function to train the network using iterative approach with proper UI updates
 function trainNetwork() {
     if (!wasmModule) {
         alert("WASM module not loaded yet!");
         return;
     }

     if (!networkBuilt) {
         alert("Please build a network first!");
         document.querySelector('.tab[data-tab="design"]').click();
         return;
     }

     // Get training parameters
     const iterations = parseInt(document.getElementById('iterations').value);
     const learningRate = parseFloat(document.getElementById('learningRate').value);

     // Show training animation
     document.getElementById('trainAnimation').style.display = 'block';

     // Disable train button during training
     document.getElementById('trainNetworkBtn').disabled = true;

     // Clear previous training info
     document.getElementById('trainingInfo').style.display = 'none';

     // Set up progress bar container if not already present
     let progressContainer = document.getElementById('trainingProgressContainer');
     if (!progressContainer) {
         progressContainer = document.createElement('div');
         progressContainer.id = 'trainingProgressContainer';
         progressContainer.className = 'progress-container';
         document.getElementById('trainAnimation').appendChild(progressContainer);

         // Create actual progress bar element
         const progressBar = document.createElement('div');
         progressBar.id = 'trainingProgressBar';
         progressBar.className = 'progress-bar';
         progressContainer.appendChild(progressBar);

         // Create text indicator
         const progressText = document.createElement('div');
         progressText.id = 'trainingProgressText';
         progressText.className = 'progress-text';
         progressContainer.appendChild(progressText);
     }

     // Get references to our progress elements
     const progressBar = document.getElementById('trainingProgressBar');
     const progressText = document.getElementById('trainingProgressText');

     // Initialize progress display
     progressBar.style.width = '0%';
     progressText.textContent = 'Starting training...';

     // Record start time
     const startTime = performance.now();

     // Initialize the trainer with the learning rate
     wasmModule.ccall('Initialize_Trainer_For_One_Step_Iteration', null, ['number'], [learningRate]);

     // Track iterations
     let currentIteration = 0;

     // Function to run one training step
     function runTrainingIteration() {
         // Check if we've completed all iterations
         if (currentIteration >= iterations) {
             // Training complete - finalize and update UI
             const endTime = performance.now();
             const trainingTime = ((endTime - startTime) / 1000).toFixed(2);
             const finalLoss = wasmModule.ccall('Get_Block_Loss', 'number', [], []);

             // Hide animation
             document.getElementById('trainAnimation').style.display = 'none';

             // Update training info display
             document.getElementById('iterationInfo').textContent = `Iterations: ${iterations}`;
             document.getElementById('learningRateInfo').textContent = `Learning Rate: ${learningRate}`;
             document.getElementById('trainingTimeInfo').textContent = `Training Time: ${trainingTime}s`;
             document.getElementById('finalLossInfo').textContent = `Final Loss: ${finalLoss.toFixed(6)}`;

             // Show training info
             document.getElementById('trainingInfo').style.display = 'block';

             // Enable train button
             document.getElementById('trainNetworkBtn').disabled = false;

             // Update results and switch to results tab
             updateResults();
             document.querySelector('.tab[data-tab="results"]').click();

             return;
         }

         // Execute one training step
         wasmModule.ccall('Train_Network_By_One_Iteration', null, [], []);

         // Get the current loss
         const currentLoss = wasmModule.ccall('Get_Block_Loss', 'number', [], []);

         // Update progress display
         const progressPercent = Math.round((currentIteration / iterations) * 100);
         progressBar.style.width = `${progressPercent}%`;
         progressText.textContent = `Progress: ${progressPercent}% (Iteration ${currentIteration+1}/${iterations}, Loss: ${currentLoss.toFixed(6)})`;

         // Increment iteration counter
         currentIteration++;

         // Use requestAnimationFrame to ensure UI updates
         // plus a small delay to allow the UI to actually render between iterations
         setTimeout(() => {
             requestAnimationFrame(runTrainingIteration);
         }, 10); // Small delay of 10ms to allow UI updates
     }

     // Start the training process
     requestAnimationFrame(runTrainingIteration);
 }

    // Function to update results after training
    function updateResults() {
    if (!wasmModule) {
    return;
}

    // Get predictions
    const predictionsArray = new Float32Array(16);
    const predictionsPtr = wasmModule._malloc(16 * 4); // Float32 = 4 bytes

    try {
    wasmModule.ccall('getPredictions', null, ['number'], [predictionsPtr]);

    // Copy from WASM memory to JavaScript array
    for (let i = 0; i < 16; i++) {
    predictionsArray[i] = wasmModule.HEAPF32[predictionsPtr / 4 + i];
}

    // Get input patterns
    const inputPatternsArray = new Float32Array(64); // 16 patterns * 4 bits
    const inputPatternsPtr = wasmModule._malloc(64 * 4); // Float32 = 4 bytes

    // Get target values
    const targetValuesArray = new Float32Array(16);
    const targetValuesPtr = wasmModule._malloc(16 * 4); // Float32 = 4 bytes

    wasmModule.ccall('getInputPatterns', null, ['number'], [inputPatternsPtr]);
    wasmModule.ccall('getTargetValues', null, ['number'], [targetValuesPtr]);

    // Copy from WASM memory to JavaScript arrays
    for (let i = 0; i < 64; i++) {
    inputPatternsArray[i] = wasmModule.HEAPF32[inputPatternsPtr / 4 + i];
}

    for (let i = 0; i < 16; i++) {
    targetValuesArray[i] = wasmModule.HEAPF32[targetValuesPtr / 4 + i];
}

    // Update results table
    updateResultsTable(inputPatternsArray, targetValuesArray, predictionsArray);

    // Clean up
    wasmModule._free(inputPatternsPtr);
    wasmModule._free(targetValuesPtr);

} finally {
    wasmModule._free(predictionsPtr);
}
}

    // Function to update the results table
    function updateResultsTable(inputPatterns, targetValues, predictions) {
    const tableBody = document.getElementById('resultsTableBody');
    tableBody.innerHTML = '';

    let correctCount = 0;

    for (let i = 0; i < 16; i++) {
    // Extract the 4 bits for this pattern
    const bits = [
    Math.round(inputPatterns[i * 4]),
    Math.round(inputPatterns[i * 4 + 1]),
    Math.round(inputPatterns[i * 4 + 2]),
    Math.round(inputPatterns[i * 4 + 3])
    ];

    // Get target and prediction
    const target = Math.round(targetValues[i]);
    const prediction = predictions[i];
    const roundedPrediction = Math.round(prediction);
    const correct = roundedPrediction === target;

    if (correct) correctCount++;

    // Create table row
    const row = document.createElement('tr');
    row.className = correct ? 'correct' : 'incorrect';

    row.innerHTML = `
                <td>${i}</td>
                <td>${bits.join('')}</td>
                <td>${target}</td>
                <td>${prediction.toFixed(4)}</td>
                <td>${roundedPrediction}</td>
                <td>${correct ? '✓' : '✗'}</td>
            `;

    tableBody.appendChild(row);
}

    // Update accuracy
    const accuracy = (correctCount / 16) * 100;
    document.getElementById('accuracyInfo').textContent = `Accuracy: ${accuracy.toFixed(2)}% (${correctCount}/16 correct)`;
}

    // Function to update network information display
    function updateNetworkInfo(blockSize, loss) {
    const networkInfo = document.getElementById('networkInfo');

    let layersInfo = hiddenLayers.map((layer, index) => {
    const activationName = activationOptions.find(opt => opt.value === layer.activation).label;
    return `<p><strong>Hidden Layer ${index + 1}:</strong> ${layer.neurons} neurons with ${activationName} activation</p>`;
}).join('');

    networkInfo.innerHTML = `
                <h3>Network Information</h3>
                <p><strong>Input Layer:</strong> 4 neurons (for 4-bit parity problem)</p>
                ${layersInfo}
                <p><strong>Output Layer:</strong> 1 neuron with Sigmoid activation</p>
                <p><strong>Loss Function:</strong> Binary Cross-Entropy</p>
                <p><strong>Total Layers:</strong> ${blockSize}</p>
                <p><strong>Initial Loss:</strong> ${loss}</p>
            `;
}

 /**
  * Neural Network Architecture Visualization
  * Uses D3.js to create an interactive visualization of the neural network
  */
 class NetworkVisualizer {
     constructor(containerId) {
         this.containerId = containerId;
         this.container = document.getElementById(containerId);

         // Default dimensions in case container isn't ready
         this.width = 800;
         this.height = 400;
         this.margin = { top: 40, right: 40, bottom: 40, left: 40 };
         this.neuronRadius = 20;
         this.colors = {
             input: '#b8b5e1',    // Light purple
             hidden: '#8ab5f0',    // Light blue
             output: '#9cefda',    // Light teal
             text: '#333333',      // Dark gray
             connection: '#cccccc' // Light gray
         };

         this.initialized = false;
         this.networkData = null;
     }

     /**
      * Initialize the SVG element safely
      */
     initSvg() {
         try {
             // Safety check for container
             this.container = document.getElementById(this.containerId);
             if (!this.container || !document.body.contains(this.container)) {
                 console.warn(`Container #${this.containerId} not found or not in DOM`);
                 return false;
             }

             // Get actual dimensions from the container
             this.width = this.container.clientWidth || this.width;

             // Clear any existing SVG
             this.container.innerHTML = '';

             // Create SVG element
             this.svg = d3.select(`#${this.containerId}`)
                 .append('svg')
                 .attr('width', this.width)
                 .attr('height', this.height);

             this.mainGroup = this.svg.append('g')
                 .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);

             // Add definitions for arrowheads
             this.svg.append('defs').append('marker')
                 .attr('id', 'arrowhead')
                 .attr('viewBox', '0 -5 10 10')
                 .attr('refX', 8)
                 .attr('refY', 0)
                 .attr('markerWidth', 4)
                 .attr('markerHeight', 4)
                 .attr('orient', 'auto')
                 .append('path')
                 .attr('d', 'M0,-5L10,0L0,5')
                 .attr('fill', this.colors.connection);

             // Create tooltip if it doesn't exist
             if (!document.querySelector('.nn-tooltip')) {
                 this.tooltip = d3.select('body').append('div')
                     .attr('class', 'nn-tooltip')
                     .style('position', 'absolute')
                     .style('visibility', 'hidden')
                     .style('background-color', 'white')
                     .style('border', '1px solid #ddd')
                     .style('border-radius', '4px')
                     .style('padding', '8px')
                     .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
                     .style('font-size', '12px')
                     .style('pointer-events', 'none')
                     .style('z-index', '1000');
             } else {
                 this.tooltip = d3.select('.nn-tooltip');
             }

             this.initialized = true;
             return true;
         } catch (error) {
             console.error('Error initializing SVG:', error);
             return false;
         }
     }

     /**
      * Render the neural network visualization
      */
     render(networkStructure) {
         try {
             // Store the network data for potential re-renders
             this.networkData = networkStructure;

             // Initialize SVG if needed
             if (!this.initialized && !this.initSvg()) {
                 console.warn('Cannot render: SVG initialization failed');
                 return;
             }

             // Safety check - reinitialize if the container has changed
             const container = document.getElementById(this.containerId);
             if (!container || this.container !== container) {
                 if (!this.initSvg()) {
                     console.warn('Cannot render: SVG reinitialization failed');
                     return;
                 }
             }

             // Clear previous visualization
             if (this.mainGroup) {
                 this.mainGroup.selectAll('*').remove();
             } else {
                 if (!this.initSvg()) {
                     console.warn('Cannot render: SVG reinitialization failed');
                     return;
                 }
             }

             // Calculate effective width and height
             const width = this.width - this.margin.left - this.margin.right;
             const height = this.height - this.margin.top - this.margin.bottom;

             // Set up layer structure
             const layers = [
                 { name: 'Input Layer', neurons: 4, activation: 'Linear' },
                 ...networkStructure.map((layer, i) => ({
                     name: `Hidden Layer ${i+1}`,
                     neurons: layer.neurons,
                     activation: this._getActivationName(layer.activation)
                 })),
                 { name: 'Output Layer', neurons: 1, activation: 'Sigmoid' }
             ];

             // Calculate horizontal spacing between layers
             const layerSpacing = width / (layers.length + 1);

             // Create groups for connections and neurons
             const connectionsGroup = this.mainGroup.append('g').attr('class', 'connections');
             const neuronsGroup = this.mainGroup.append('g').attr('class', 'neurons');

             // Draw connections first (behind neurons)
             this._drawConnections(connectionsGroup, layers, layerSpacing, height);

             // Draw neurons
             this._drawNeurons(neuronsGroup, layers, layerSpacing, height);

             // Add layer labels
             this._addLayerLabels(layers, layerSpacing, height);

         } catch (error) {
             console.error('Error rendering network visualization:', error);
         }
     }

     /**
      * Draw neurons for each layer
      */
     _drawNeurons(group, layers, layerSpacing, height) {
         layers.forEach((layer, layerIndex) => {
             // Calculate vertical spacing for neurons in this layer
             const maxNeurons = layer.neurons;
             const neuronSpacing = Math.min(height / (maxNeurons + 1), 50);
             const layerHeight = neuronSpacing * (maxNeurons - 1);
             const startY = (height - layerHeight) / 2;

             // Determine neuron color based on layer type
             let color;
             if (layerIndex === 0) {
                 color = this.colors.input;
             } else if (layerIndex === layers.length - 1) {
                 color = this.colors.output;
             } else {
                 color = this.colors.hidden;
             }

             // Create neuron circles with data
             const neurons = group.selectAll(`.neuron-layer-${layerIndex}`)
                 .data(Array(layer.neurons).fill().map((_, i) => ({
                     layer: layerIndex,
                     index: i,
                     name: layerIndex === 0 ? `Input ${i+1}` :
                         layerIndex === layers.length - 1 ? 'Output' : `Neuron ${i+1}`,
                     layerName: layer.name,
                     activation: layer.activation
                 })))
                 .enter()
                 .append('circle')
                 .attr('class', `neuron-layer-${layerIndex}`)
                 .attr('cx', layerSpacing * (layerIndex + 1))
                 .attr('cy', (d, i) => startY + i * neuronSpacing)
                 .attr('r', this.neuronRadius)
                 .attr('fill', color)
                 .attr('stroke', '#666')
                 .attr('stroke-width', 1)
                 .style('cursor', 'pointer');

             // Add interactivity
             if (this.tooltip) {
                 neurons
                     .on('mouseover', (event, d) => {
                         this.tooltip
                             .html(`<strong>${d.name}</strong><br/>
                                  Layer: ${d.layerName}<br/>
                                  ${layerIndex > 0 ? `Activation: ${d.activation}` : ''}`)
                             .style('visibility', 'visible')
                             .style('left', `${event.pageX + 10}px`)
                             .style('top', `${event.pageY + 10}px`);

                         // Highlight the neuron
                         d3.select(event.currentTarget)
                             .attr('stroke', '#000')
                             .attr('stroke-width', 2);
                     })
                     .on('mousemove', (event) => {
                         this.tooltip
                             .style('left', `${event.pageX + 10}px`)
                             .style('top', `${event.pageY + 10}px`);
                     })
                     .on('mouseout', (event) => {
                         this.tooltip.style('visibility', 'hidden');
                         d3.select(event.currentTarget)
                             .attr('stroke', '#666')
                             .attr('stroke-width', 1);
                     });
             }
         });
     }

     /**
      * Draw connections between neurons
      */
     _drawConnections(group, layers, layerSpacing, height) {
         for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
             const sourceLayer = layers[layerIndex];
             const targetLayer = layers[layerIndex + 1];

             // Calculate vertical spacing
             const sourceNeuronSpacing = Math.min(height / (sourceLayer.neurons + 1), 50);
             const sourceLayerHeight = sourceNeuronSpacing * (sourceLayer.neurons - 1);
             const sourceStartY = (height - sourceLayerHeight) / 2;

             const targetNeuronSpacing = Math.min(height / (targetLayer.neurons + 1), 50);
             const targetLayerHeight = targetNeuronSpacing * (targetLayer.neurons - 1);
             const targetStartY = (height - targetLayerHeight) / 2;

             // Create connections
             for (let i = 0; i < sourceLayer.neurons; i++) {
                 for (let j = 0; j < targetLayer.neurons; j++) {
                     group.append('line')
                         .attr('x1', layerSpacing * (layerIndex + 1))
                         .attr('y1', sourceStartY + i * sourceNeuronSpacing)
                         .attr('x2', layerSpacing * (layerIndex + 2))
                         .attr('y2', targetStartY + j * targetNeuronSpacing)
                         .attr('stroke', this.colors.connection)
                         .attr('stroke-width', 1)
                         .attr('stroke-opacity', 0.5)
                         .attr('marker-end', 'url(#arrowhead)');
                 }
             }
         }
     }

     /**
      * Add labels for each layer
      */
     _addLayerLabels(layers, layerSpacing, height) {
         layers.forEach((layer, layerIndex) => {
             this.mainGroup.append('text')
                 .attr('x', layerSpacing * (layerIndex + 1))
                 .attr('y', height + 30)
                 .attr('text-anchor', 'middle')
                 .attr('font-size', '14px')
                 .attr('font-weight', 'bold')
                 .attr('fill', this.colors.text)
                 .text(layer.name);
         });
     }

     /**
      * Convert activation ID to name
      */
     _getActivationName(activationId) {
         const activations = {
             0: 'ReLU',
             1: 'Sigmoid',
             2: 'Tanh',
             3: 'Leaky ReLU',
             4: 'Swish',
             5: 'Linear'
         };
         return activations[activationId] || 'Unknown';
     }

     /**
      * Safely resize the visualization
      */
     resize() {
         try {
             // Safety check for container
             const container = document.getElementById(this.containerId);
             if (!container || !document.body.contains(container)) {
                 return;
             }

             // Get new width and update if changed
             const newWidth = container.clientWidth || this.width;
             if (newWidth !== this.width && this.svg) {
                 this.width = newWidth;
                 this.svg.attr('width', this.width);

                 // Re-render with stored data if available
                 if (this.networkData) {
                     this.render(this.networkData);
                 }
             }
         } catch (error) {
             console.warn('Error during resize:', error);
         }
     }
 }

 // Initialize the visualization system
 document.addEventListener('DOMContentLoaded', function() {
     // Wait a bit to ensure DOM is fully loaded
     setTimeout(() => {
         try {
             window.networkVisualizer = new NetworkVisualizer('network-visualization');

             // Add resize handler
             window.addEventListener('resize', () => {
                 if (window.networkVisualizer) {
                     try {
                         window.networkVisualizer.resize();
                     } catch (e) {
                         console.warn('Resize error:', e);
                     }
                 }
             });

             // Handle tab changes
             const designTab = document.querySelector('.tab[data-tab="design"]');
             if (designTab) {
                 designTab.addEventListener('click', () => {
                     setTimeout(() => {
                         if (window.networkVisualizer && window.networkBuilt && window.hiddenLayers) {
                             window.networkVisualizer.render(window.hiddenLayers);
                         }
                     }, 50);
                 });
             }
         } catch (error) {
             console.error('Error initializing network visualizer:', error);
         }
     }, 100);
 });

