import numpy as np

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]


weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]
biases2 = [-1, 2, -0.5]

layer_outputs = []


for input_vector in inputs:
    neuron_outputs = [] #saida dos neuronios
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(input_vector, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        neuron_outputs.append(neuron_output)
    layer_outputs.append(neuron_outputs)

print("Manual:", layer_outputs)


layer1_outputs = np.dot(inputs, np.array(weights).T)+ biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer1_outputs)
print(layer2_outputs)



#em um mlp , o bias sube e desce a linha da funcao e o weight gira ela