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

    // Add a progress indicator if not already present
    let progressIndicator = document.getElementById('trainingProgress');
    if (!progressIndicator) {
    progressIndicator = document.createElement('div');
    progressIndicator.id = 'trainingProgress';
    progressIndicator.className = 'training-progress';
    document.getElementById('trainAnimation').appendChild(progressIndicator);
}
    progressIndicator.textContent = 'Starting training...';

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

    // Update progress indicator
    const progressPercent = Math.round((currentIteration / iterations) * 100);
    progressIndicator.textContent = `Progress: ${progressPercent}% (Iteration ${currentIteration+1}/${iterations}, Loss: ${currentLoss.toFixed(6)})`;

    // Increment iteration counter
    currentIteration++;

    // Schedule the next iteration with a short delay to allow UI updates
    setTimeout(runTrainingIteration, 0);
}

    // Start the training process
    runTrainingIteration();
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
