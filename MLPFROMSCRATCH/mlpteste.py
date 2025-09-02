import nnfs.datasets
import numpy as np
import nnfs
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#https://cs231n.github.io/neural-networks-case-study/
nnfs.init( )
np.random.seed(0)
from nnfs.datasets import spiral_data
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    def backward(self,dvalues):
         self.dweights = np.dot(self.inputs.T, dvalues)
         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
         self.dinputs = np.dot(dvalues, self.weights.T) 
         
class Activationrelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #funcao ReLu
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0   #derivada da relu

class Activation_softmax:
    def forward(self,inputs): #inputs como sempre e o vetor de resultados do layer anterior , no caso o ReLu
        # calcula o exponencial com o numero de euler ( antes subtrai o maior elemento)
        exp_values = np.exp(inputs - np.max(inputs, axis = 1 , keepdims= 1)) #para evitar numeros muito grandes / keepdims é para manter a dimensão do array após a divisão
        probabilities = exp_values / np.sum(exp_values, axis = 1 , keepdims = 1 ) #normalização, o axis é para soma ser feita linha por linha (axis = 0 ) causaria uma soma por colunas
        self.output = probabilities
class loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) #esse np.mean calcula a media aritimetica
        return data_loss
   
class LossCategoricalCrossEntropy(loss):
    
   
    def forward(self,y_pred,y_true): #basicamente o y_pred vai ser o resultado do ultimo layer do mlp , ou seja do softmax e o y vai ser os valores reais
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1-1e-7) #se for um valor menor que o primeiro, ele vira ele, ou seja , 
        #algo muito próximo de 0. e se for maior que o da direita , ele vira o da direita
        if len (y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true] #valores normais e pega os valores inferidos para a classe correta 
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped* y_true, axis = 1) # se for one hot encoding    
             # y_pred_clipped = basicamente os resultados da softmax so que sem os 0s , y_true = valor real das coisas
        negative_log_likelihoods = -np.log(correct_confidences)  #retorna a loss geral , basicamente
        return negative_log_likelihoods
                                       
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        #se os valores estiverem em one hot encoding
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy() # copia os valores para nao alterar os atuais
        self.dinputs[range(samples), y_true] -= 1   # para pegar as probabilidades e calcula o gradiente em relacao a relu
        self.dinputs = self.dinputs / samples 

class InicializarRedeNeural:
   def treinar(self):
    X, y = x_train, y_train  
    batch_size = 32 # numero de exemplos de cada batche 
    num_batches = X.shape[0] // batch_size #numero total de exeplos do mnist dividido pela quantidade do numero de batches, ou seja, 1000//32 = 31 batches ( acho q esse é o resultado) 
    layer1 = Layer_dense(784, 128)
    activation1 = Activationrelu()                #deixando claro que a quantidade de neuronios ou layers e entradas vai depender do seu problema, 
    layer2 = Layer_dense(128, 64)                 #ex: o mnist tem 28x28 pixels , ai meti 784 entradas , uma pra cada
    activation2 = Activationrelu()
    layer3 = Layer_dense(64, 10)
    activation3 = Activation_softmax()
    loss_func = LossCategoricalCrossEntropy()

    
    epochs = int(input("Digite a quantidade de épocas: "))

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

      
        for batch in range(num_batches): # vai ver todas as batches
            
            batch_start = batch * batch_size # encontra o comeco da batch
            batch_end = batch_start + batch_size #encontra o final dela
            X_batch = X[batch_start:batch_end]  #pega do elemento de comeco ate o final 
            y_batch = y[batch_start:batch_end]  #pega o começo até o gfinal
            
            #de resto apenas troquei o x_train pelo x_bATCH QUE CONTEM OS ELEMENTOS DO X TRAIN EM MENOR ESCALA
            layer1.forward(X_batch)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)            #feedfoward
            layer3.forward(activation2.output)
            activation3.forward(layer3.output)

           
            loss = loss_func.calculate(activation3.output, y_batch)
            predictions = np.argmax(activation3.output, axis=1)
            accuracy = np.mean(predictions == y_batch)

          
            epoch_loss += loss   #somo as loss de cada batch
            epoch_accuracy += accuracy    #faço a msma coisa cm a acc

         
            loss_func.backward(activation3.output, y_batch)
            layer3.backward(loss_func.dinputs)   # encontrei os dw do layer3 / layer3 dinputs = gradiente da ultima camada vezes o peso transposto dos seus neuronios                 #basicamente faço o backporpagation
            activation2.backward(layer3.dinputs) #chamo   
            layer2.backward(activation2.dinputs)
            activation1.backward(layer2.dinputs)
            layer1.backward(activation1.dinputs)

          
            learning_rate = 0.1
            layer1.weights -= learning_rate * layer1.dweights        #so atualizo os pesos cm os o resultado da derivada do peso pela loss
            layer1.biases -= learning_rate * layer1.dbiases
            layer2.weights -= learning_rate * layer2.dweights
            layer2.biases -= learning_rate * layer2.dbiases
            layer3.weights -= learning_rate * layer3.dweights
            layer3.biases -= learning_rate * layer3.dbiases

      
        epoch_loss /= num_batches # FAZER UMA MEDIA ENTRE as loss
        epoch_accuracy /= num_batches # fazer uma media entre as accs
        print(f'Época {epoch+1}, Loss: {epoch_loss:.4f}, Acurácia: {epoch_accuracy:.4f}')


            
        
redenaural = InicializarRedeNeural()
redenaural.treinar()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
X,y = spiral_data(samples = 100 , classes =3)
dense1 = Layer_dense(2,3)
activation1 = Activationrelu()
dense2 = Layer_dense(3,3)
activation2 = Activation_softmax()


dense1.forward(X) # passa pelo primeiro layer resultado vezes o peso mais o bias

activation1.forward(dense1.output) # apos isso , os resultados passam pela função de ativação ReLu ( para eliminar os valores negativos )
#print(dense1.output)
dense2.forward(activation1.output) # pega a saida da relu e faz a multiplicacao do peso mais os bias
#print(dense2.output)
activation2.forward(dense2.output) # por ultimo, passa esse resultado para a softmax transformar em estatistica 
#Converte os valores em uma distribuição de probabilidades (valores entre 0 e 1 que somam 1 por linha).
print(activation2.output[:5])

loss_function = LossCategoricalCrossEntropy() 
loss = loss_function.calculate(activation2.output, y )
print("loss", loss)

layer1 = Layer_dense(2, 5)  # number of inputs/ features , number of neurons
activation1 = Activationrelu()
layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
'''

 