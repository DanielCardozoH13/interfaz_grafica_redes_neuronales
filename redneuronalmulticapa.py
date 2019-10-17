import numpy as np
from time import time

def sigmoide(x):
    return 1/(1 + np.exp(-x))

def sigmoide_derivado(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tangente(x):
    return np.tanh(x)

def tangente_derivada(x):
    return 1 - x**2

class RedNeuronalMulticapa():
    def __init__(self):
    	#Iniciarlizar pesos
    	self.pesos = []
    	self.deltas = []

    def run(self, capas, activacion='tangente'):
        if activacion == 'sigmoide':
            self.activacion = sigmoide
            self.activacion_prima = sigmoide_derivado
        elif activacion == 'tangente':
            self.activacion = tangente
            self.activacion_prima = tangente_derivada

        self.pesos = []
        self.deltas = []
            
        
        
        for i in range(1, len(capas) -1):
            r = 2 * np.random.random((capas[i-1] + 1, capas[i] + 1)) -1
            self.pesos.append(r)
        
        #asignar aleatorios a la capa de salida
        r = 2 * np.random.random((capas[i] + 1, capas[i + 1])) - 1
        self.pesos.append(r)
    def ajuste(self, X, y, factor_aprendizaje = 0.2, epocas = 1000):
        self.start_time = time()
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis = 1)
        
        for k in range(epocas):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            
            for l in range(len(self.pesos)):
                dot_value = np.dot(a[l], self.pesos[l])
                activacion = self.activacion(dot_value)
                a.append(activacion)
                
            #Calculo la diferencia entre la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activacion_prima(a[-1])]
            
            #Empezamos en la segunda capa hasta la ultima
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.pesos[l].T) * self.activacion_prima(a[l]))
            self.deltas.append(deltas)
            
            #invertir
            deltas.reverse()
            
            #Backpropagation
            for i in range(len(self.pesos)):
                capa = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.pesos[i] += factor_aprendizaje * capa.T.dot(delta)
            
            if k % 10000 == 0: print('epocas:', k)
        self.elapsed_time = time() - self.start_time
                
    def predecir(self, x):
        unos = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis = 0)
        for l in range(0, len(self.pesos)):
            a = self.activacion(np.dot(a, self.pesos[l]))
        return a

    def obtener_pesos(self):
    	return self.pesos
    
    def obtener_deltas(self):
        return self.deltas    

def main():
    nn = RedNeuronalMulticapa()
    estructura_capas = [2,3,2]
    nn.run(estructura_capas, activacion="tangente")
    X = np.array([[0,0], #sin obstaculos
                 [0,1], #sin obstaculos
                 [0,-1],  #sin obstaculos
                 [0.5, 1], #obstaculo a la derecha
                 [0.5, -1], #obstaculo a la izquierda
                 [1, 1], #demasiado cerca a la derecha
                 [1,-1]]) #demasiado cerca a la izquierda

    y = np.array([[0,1], #avanzar
                 [0,1],
                 [0,1],
                 [-1,1], #giro izquierda
                 [1,1], #giro derecha
                 [0,-1],  #retroceder
                 [0,-1],]) #retroceder

    nn.ajuste(X, y, factor_aprendizaje = 0.03, epocas = 15000)
    print('tiempo transcurrido en entrenamiento: {}'.format(nn.elapsed_time))

    index = 0
    for e in X:
        print("X: ", e, "y: ", y[index], "Red: ", nn.predecir(e))
        index = index + 1
if __name__ == '__main__':
    main()