from numpy import exp, array, random, dot, append
from time import time


class RedNeuronalSimple():
    def __init__(self):
        self.pesos_sinapticos = list()
        self.errores = list()
    
    def __sigmoide(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoide_derivado(self, x):
        return x * (1 - x)
    
    def __entrenamiento(self,entradas,salidas,numero_iteraciones):
        self.start_time = time()
        entradas = array(entradas)
        if type(salidas) is list:
        	salidas = [salidas]
        	salidas = array(salidas).T
        for i in range(numero_iteraciones):
            salida = self.prediccion(entradas)
            error = salidas - salida
            ajuste = dot(entradas.T, error * self.__sigmoide_derivado(salida))
            self.errores.append(error)
            self.pesos_sinapticos += ajuste
        self.elapsed_time = time() - self.start_time
            
    def prediccion(self,entrada):
        return self.__sigmoide(dot(entrada, self.pesos_sinapticos))

    def run(self, cantidad_entradas,entradas,salidas,numero_iteraciones):
    	self.pesos_sinapticos = 2 * random.random((cantidad_entradas,1)) - 1
    	self.__entrenamiento(entradas,salidas,numero_iteraciones)

    def obtener_pesos(self):
    	return self.pesos_sinapticos

    def obtener_errores(self):
    	return self.errores

def main():
    red_neuronal = RedNeuronalSimple()
    entradas = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    salidas = array([[0,1,1,0]]).T
    # print(salidas)
    red_neuronal.run(cantidad_entradas = 3, entradas=entradas, salidas=salidas,numero_iteraciones=1)
    print("tiempo transcurrido en entrenamiento. {}".format(red_neuronal.elapsed_time))

    entrada_prueba = array([1,0,0])
    print("prediccion para la entrada {} es {}".format(entrada_prueba, red_neuronal.prediccion(entrada_prueba)))


if __name__ == '__main__':
    main()