from random import choice, randrange
from numpy import array, dot, random
from time import time

class Perceptron():
	def __init__(self):
		super(Perceptron, self).__init__()
		self.cantidad_entradas = int()
		self.pesos = list()
		self.bahia = float()
		self.epocas = int()
		self.entradas_entrenamiento = list()
		self.salidas_entrenamiento = list()
		self.entrenamiento = list()
		self.error = float()
		self.esperados = list()
		self.errores = list()
		
	def run(self, cantidad_entradas, pesos, bahia, epocas, entradas_entrenamiento, salidas_entrenamiento):
		self.cantidad_entradas = cantidad_entradas
		self.pesos = pesos
		self.bahia = bahia
		self.epocas = epocas
		self.entradas_entrenamiento = entradas_entrenamiento
		self.salidas_entrenamiento = salidas_entrenamiento

		self.__cargar_set_entrenamiento()
		self.__cargar_pesos_sinapticos()
		self.__entrenamiento()

	

	def __funcion_activiacion(self, x):
		return 0 if x < 0 else 1

	def __cargar_set_entrenamiento(self):
		if (self.entradas_entrenamiento and self.salidas_entrenamiento) and len(self.entradas_entrenamiento) == len(self.salidas_entrenamiento):
			for i,salida in enumerate(self.salidas_entrenamiento):
				self.entrenamiento.append((array(self.entradas_entrenamiento[i]), salida))
		else:
			for i in range(self.cantidad_entradas):
				self.entrenamiento.append((random.rand(self.cantidad_entradas), randrange(1)))

	def __cargar_pesos_sinapticos(self):
		if type(self.pesos) is int or len(self.pesos) != self.cantidad_entradas:
			# self.pesos = random.rand(self.cantidad_entradas)
			self.pesos = 2 * random.random((self.cantidad_entradas,1)) - 1
			print(len(self.pesos))

	def __entrenamiento(self):
		self.start_time = time()
		for i in range(self.epocas):
		    x, esperado = choice(self.entrenamiento)
		    resultado = dot(self.pesos, x)
		    self.error = esperado - self.__funcion_activiacion(resultado)
		    self.esperados.append(esperado)
		    self.errores.append(self.error)
		    #ajuste
		    self.pesos += self.bahia * self.error * x
		self.elapsed_time = time() - self.start_time

	def prediccion(self, entradas):
		entrada_neta = dot(self.pesos, entradas)
		resultado = self.__funcion_activiacion(entrada_neta)
		return resultado
		    

	def obtener_pesos(self):
		return self.pesos

	def obtener_errores(self):
		return self.errores

	def obtener_esperados(self):
		return self.esperados

def main():
	red_perceptron = Perceptron()
	"""
	entrenamiento = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
	]"""
	epocas = 50000
	bahias = 0.2
	w = random.rand(3)
	entradas = [
		(array([0,0,1])),
    	(array([0,1,1])),
    	(array([1,0,1])),
    	(array([1,1,1])),
	]
	salidas = [0,1,1,1]


	red_perceptron.run(cantidad_entradas=3, pesos=w, bahia = bahias, epocas = epocas, entradas_entrenamiento = entradas, salidas_entrenamiento = salidas)
	print('tiempo transcurrido en entrenamiento: {}'.format(red_perceptron.elapsed_time))

	entrada=[0,0,1]

	print("Entrada {}".format(entrada))
	print("salida: {}".format(red_perceptron.prediccion(entrada)))

if __name__ == '__main__':
	main()