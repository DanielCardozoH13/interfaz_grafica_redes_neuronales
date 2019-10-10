from tkinter import *
from tkinter import ttk
from numpy import random, array
import perceptron, redneuronalsimple, redneuronalmulticapa, matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Graficadora():

    def __init__(self, perceptron, rns, rnm):
        self.ventana = Tk()
        self.ventana.title("Graficador de redes neuronales")
        self.ventana.geometry("680x600")
        self.ventana.resizable(width=False, height=False)

        #variables utilizadas en el perceptrón
        self.cantidad_entradas = IntVar()
        self.cantidad_entradas.set(2)
        self.pesos_sin = list()
        self.bahia = IntVar()
        self.cantidad_datos_entrenamiento = IntVar()
        self.cantidad_datos_entrenamiento.set(4)
        self.entradas_set = list()
        self.salidas_set = list()
        self.red_perceptron = perceptron
        self.resultado_prediccion_perceptron = ""
        self.entradas_prediccion = list()

        #variables utilizadas en la rns
        self.cantidad_entradas_rns = IntVar()
        self.cantidad_entradas_rns.set(2)
        self.pesos_sin_rns = list()
        self.cantidad_datos_entrenamiento_rns = IntVar()
        self.cantidad_datos_entrenamiento_rns.set(4)
        self.entradas_set_rns = list()
        self.salidas_set_rns = list()
        self.red_neuronal_simple = rns
        self.resultado_prediccion_rns = ""
        self.entradas_prediccion_rns = list()

        #varibles utilizada en la rnm
        self.entradas_set_rnm = list()
        self.salidas_set_rnm = list()
        self.red_neuronal_multicapa = rnm
        self.entradas_prediccion_rnm = list()
        self.resultado_prediccion_rnm = ""

        self.run()

    def run(self):
        ##--configuracion de canvas para colocar scrollbar--##
        scrollbar = Scrollbar(self.ventana)
        c = Canvas(self.ventana, yscrollcommand=scrollbar.set)
        scrollbar.config(command=c.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.frame_principal = Frame(c)
        c.pack(side=LEFT, fill=BOTH, expand=True)
        c.create_window(0, 0, window=self.frame_principal, anchor='nw')
        ##------##

        self.cuaderno = ttk.Notebook(self.frame_principal)
        self.cuaderno.pack(fill=BOTH, expand=True)

        self.pagina1 = ttk.Frame(self.cuaderno)
        self.cuaderno.add(self.pagina1, text="Perceptrón")

        self.pagina2 = ttk.Frame(self.cuaderno)
        self.cuaderno.add(self.pagina2, text="Red N. Simple")

        self.pagina3 = ttk.Frame(self.cuaderno)
        self.cuaderno.add(self.pagina3, text="Red N. Multicapa")

        self.__frame_parametros_perceptron()
        self.__frame_parametros_rns()
        self.__frame_parametros_rnm()

        self.cuaderno.pack(fill=BOTH, expand=True)

        # --configuracion para colocar scrollbar--##
        self.ventana.update()
        c.config(scrollregion=c.bbox("all"))
        ##------##

        self.ventana.mainloop()

    def __frame_parametros_perceptron(self):
        try:
            del self.entradas_prediccion[:]
            self.labelframe_grafica.grid_remove()
        except:
            pass
        self.labelframe_parametros = ttk.LabelFrame(self.pagina1, text="Parametros Perceptrón:")
        self.labelframe_parametros.grid(column=0, row=0, ipady=1880)
        self.__frame_entradas()
        self.__frame_pesos()
        self.__frame_bahia()
        self.__frame_entrenamiento()
        self.__frame_boton_accion()

    def __frame_entradas(self):
        frame_entradas = ttk.LabelFrame(self.labelframe_parametros, text="Cantidad Entradas:")
        frame_entradas.grid(column=0, row=1, ipadx=101, pady=10, padx=0, sticky=NW)
        label_cant_entradas = ttk.Label(frame_entradas,
                                        text="Seleccione la cantidad de entradas que tendra el perceptrón:")
        label_cant_entradas.grid(column=0, row=0, padx=4, pady=4)
        ttk.Radiobutton(frame_entradas, text='2', variable=self.cantidad_entradas, value=2,
                        command=lambda: self.__cambio_cant_entrada()).grid(column=1, row=1)
        ttk.Radiobutton(frame_entradas, text='3', variable=self.cantidad_entradas, value=3,
                        command=lambda: self.__cambio_cant_entrada()).grid(column=2, row=1)
        ttk.Radiobutton(frame_entradas, text='4', variable=self.cantidad_entradas, value=4,
                        command=lambda: self.__cambio_cant_entrada()).grid(column=3, row=1)
        ttk.Radiobutton(frame_entradas, text='5', variable=self.cantidad_entradas, value=5,
                        command=lambda: self.__cambio_cant_entrada()).grid(column=4, row=1)

    def __cambio_cant_entrada(self):
        del self.pesos_sin[:]
        self.frame_pesos.destroy()
        self.__frame_pesos()

        self.__cambio_cant_datos_entren()

    def __frame_pesos(self):
        self.frame_pesos = ttk.LabelFrame(self.labelframe_parametros, text="Pesos Sinápticos:")
        self.frame_pesos.grid(column=0, row=2, ipadx=120, sticky=NW)
        self.label = ttk.Label(self.frame_pesos,
                               text="Por defecto se generan pesos aleatorios o ingrese\n sus propios pesos sinápticos iniciales.")
        self.label.grid(column=0, row=0, padx=5, pady=10)
        self.pesos_sin = self.__campos(self.frame_pesos, "Peso", self.cantidad_entradas.get())

    def __campos(self, frame, text_variable, cantidad_campos):
        cantidad = int(cantidad_campos)
        variable = []
        for i in range(cantidad):
            text_label = "{} {}:".format(text_variable, i + 1)
            label = ttk.Label(frame, text=text_label)
            label.grid(column=0, row=i + 1)
            en = ttk.Entry(frame)
            en.grid(column=1, row=i + 1, padx=4, pady=4)
            variable.append(en)
        return variable

    def __frame_bahia(self):
        self.frame_bahia = ttk.LabelFrame(self.labelframe_parametros, text="Bahia:")
        self.frame_bahia.grid(column=0, row=3, ipadx=120, sticky=NW)

        label_bahia = ttk.Label(self.frame_bahia, text="Bahia: ")
        label_bahia.grid(column=0, row=0, padx=117, pady=4)
        self.bahia = ttk.Entry(self.frame_bahia)
        self.bahia.grid(column=1, row=0, padx=4, pady=4)
       
    def __frame_entrenamiento(self):
        self.frame_entrenamiento = ttk.LabelFrame(self.labelframe_parametros, text="Set de entrenamiento")
        self.frame_entrenamiento.grid(column=0, row=4, ipadx=10, sticky=NW)

        label_cant_epocas = ttk.Label(self.frame_entrenamiento, text="Seleccione la cantidad epocas (repeticiones):")
        label_cant_epocas.grid(column=0, row=0, padx=4, pady=4)
        self.epocas = ttk.Entry(self.frame_entrenamiento)
        self.epocas.grid(column=0, row=1, padx=4, pady=4)

        label_cant_datos_entrenamientos = ttk.Label(self.frame_entrenamiento, text="Seleccione la cantidad de datos de entrenamiento:")
        label_cant_datos_entrenamientos.grid(column=0, row=2, padx=4, pady=4)
        ttk.Radiobutton(self.frame_entrenamiento, text='4', variable=self.cantidad_datos_entrenamiento, value=4,
                        command=lambda: self.__cambio_cant_datos_entren()).grid(column=0, row=3)
        ttk.Radiobutton(self.frame_entrenamiento, text='10', variable=self.cantidad_datos_entrenamiento, value=10,
                        command=lambda: self.__cambio_cant_datos_entren()).grid(column=0, row=4)
        ttk.Radiobutton(self.frame_entrenamiento, text='15', variable=self.cantidad_datos_entrenamiento, value=15,
                        command=lambda: self.__cambio_cant_datos_entren()).grid(column=0, row=5)
        ttk.Radiobutton(self.frame_entrenamiento, text='20', variable=self.cantidad_datos_entrenamiento, value=20,
                        command=lambda: self.__cambio_cant_datos_entren()).grid(column=0, row=6)

        label_entrenameinto = ttk.Label(self.frame_entrenamiento,
                                        text="Ingrese las entradas y salidas esperadas para el entrenamiento:",
                                        justify=LEFT)
        label_entrenameinto.grid(column=0, row=7, pady=7)

        self.__set_entrenamiento()

    def __set_entrenamiento(self):
        cantidad_datos = self.cantidad_datos_entrenamiento.get()
        cantidad_entradas = self.cantidad_entradas.get()
        row = 7  # indica la fila desde donde inicia
        for i in range(cantidad_datos):
            row += i
            subframe_entrenamiento = ttk.LabelFrame(self.frame_entrenamiento, text="set " + str(i + 1))
            subframe_entrenamiento.grid(column=0, row=row, ipadx=100, padx=30, pady=7, sticky=NW)
            lista_entradas = self.__campos_entrenamiento(subframe_entrenamiento, self.salidas_set, row, cantidad_entradas)
            self.entradas_set.append(lista_entradas)
            del lista_entradas[:cantidad_datos:-1]

    def __campos_entrenamiento(self, frame, variable_salidas, fila, cantidad_entradas, cantidad_salidas = 1):
        variable_entradas = list()
        cantidad_entr = int(cantidad_entradas)
        for i in range(cantidad_entr):
            text_label = "Entrada {}:".format(i + 1)
            label = ttk.Label(frame, text=text_label)
            label.grid(column=0, row=i + 1 + fila)
            entry = ttk.Entry(frame)
            entry.grid(column=1, row=i + 1 + fila, padx=4, pady=7)
            variable_entradas.append(entry)
            if i == 0:
                if int(cantidad_salidas) == 1:
                    text_label = "salida: "
                    label = ttk.Label(frame, text=text_label)
                    label.grid(column=2, row=i + 1 + fila)

                    en = ttk.Entry(frame)
                    en.grid(column=3, row=i + 1 + fila, padx=4, pady=7)
                    variable_salidas.append(en)
                else:
                    column_label = 2
                    column_entry = 3
                    sub_variable_lsitas = []
                    for salida in range(int(cantidad_salidas)):
                        label = ttk.Label(frame, text="Salida {} =".format(salida + 1))
                        label.grid(column=column_label + salida, row=i + 1 + fila)

                        en = ttk.Entry(frame, justify=LEFT, width=6)
                        en.grid(column=column_entry + salida, row=i + 1 + fila, padx=4, pady=7)
                        sub_variable_lsitas.append(en)
                        column_label += 1
                        column_entry += 1
                    variable_salidas.append(sub_variable_lsitas)
        return variable_entradas

    def __cambio_cant_datos_entren(self):
        del self.entradas_set[:]
        del self.salidas_set[:]
        self.frame_entrenamiento.destroy()
        self.__frame_entrenamiento()

    def __frame_boton_accion(self):
        frame_boton_accion = ttk.LabelFrame(self.labelframe_parametros, text="Acción")
        frame_boton_accion.grid(column=0, row=5, ipadx=270, sticky=NW)

        boton_inicion = ttk.Button(frame_boton_accion, text='Calcular', state=NORMAL,
                                   command=lambda: self.__crear_neurona())
        boton_inicion.pack(side=TOP, padx=16, pady=15)

    def __crear_neurona(self):
        valida_campo = lambda x: float(x) if len(x) > 0 else 0
        cantidad_entradas = int(self.cantidad_entradas.get())
        pesos_sinapticos = self.__cargar_arreglos(self.pesos_sin, self.cantidad_entradas.get(), default_cero=False)
        bahia = valida_campo(self.bahia.get())
        epocas = int(valida_campo(self.epocas.get()))

        entradas_entrenamiento = list()
        for lista in self.entradas_set:
            result = self.__cargar_arreglos(lista, self.cantidad_datos_entrenamiento.get())
            entradas_entrenamiento.append(result)

        salidas_esperadas = self.__cargar_arreglos(self.salidas_set, self.cantidad_datos_entrenamiento.get())

        self.red_perceptron.run(cantidad_entradas=cantidad_entradas, pesos=pesos_sinapticos, bahia=bahia, epocas=epocas,
                                entradas_entrenamiento=entradas_entrenamiento, salidas_entrenamiento=salidas_esperadas)

        # se esconde el frame que contine el formulario de parametros del percetrón
        # se crea frame que contine formuario para realizar predicción y gráfica
        self.labelframe_parametros.grid_forget()
        self.__frame_grafica_perceptron()

    def __cargar_arreglos(self, lista, tamaño, default_cero=True):
        lista_convertida = list()
        if isinstance(lista, list):
            for entrada in lista:
                if entrada.get() == '':
                    if default_cero:
                        entrada.insert(0, 0)
                    else:
                        entrada.insert(0, random.rand() - 1)
                lista_convertida.append(float(entrada.get()))
        else:
            if lista.get() == "":
                if default_cero:
                    lista_convertida = 0.0
                else:
                    lista_convertida = random.rand() - 1
        return lista_convertida

    def __frame_grafica_perceptron(self):
        self.labelframe_grafica = ttk.LabelFrame(self.pagina1, text="Perceptrón:")
        self.labelframe_grafica.grid(column=0, row=0)
        self.__frame_perceptron_calculado()
        self.__frame_plot()
        self.__frame_prediciones()
        self.__frame_boton_accion_grafica()

    def __frame_perceptron_calculado(self):
        frame_perceptron_calculado = ttk.LabelFrame(self.labelframe_grafica, text="Pesos calculados")
        frame_perceptron_calculado.grid(column=0, row=0, ipadx=214, sticky=NW)
        
        pesos_calculados = self.red_perceptron.obtener_pesos() 
        texto = "Pesos calculados por la Red Perceptrón:\n\n"
        for i,peso in enumerate(pesos_calculados):
            texto += ("peso {} = {} \n".format(i, round(peso, 5)))
        label = ttk.Label(frame_perceptron_calculado, text=texto)
        label.grid(column = 0, row = 0, pady = 10, padx = 10)
       
    def __frame_plot(self):
        frame_plot = ttk.LabelFrame(self.labelframe_grafica, text="Gráfica Aprendizaje Perceptrón")
        frame_plot.grid(column=0, row=1, ipadx=48, sticky=NW)
        errores = self.red_perceptron.obtener_errores()
        esperados = self.red_perceptron.obtener_esperados()

        fig = Figure(figsize=(5.5,4.5))
        a = fig.add_subplot(111)
        a.plot(errores, color='red', label="Errores")
        a.plot(esperados, color='blue', label="Esperado")
        a.set_title ("Aprendizaje del Perceptrón", fontsize=10)
        a.set_ylabel("Valores", fontsize=12)
        a.set_xlabel("Epocas", fontsize=12)
        a.legend()

        self.canvas1 = FigureCanvasTkAgg(fig, master=frame_plot)
        self.canvas1.get_tk_widget().pack()
        self.canvas1.draw()

    def __frame_prediciones(self):
        self.frame_prediciones = ttk.LabelFrame(self.labelframe_grafica, text="Predicciones")
        self.frame_prediciones.grid(column=0, row=2, ipadx=228, sticky=NW)

        self.entradas_prediccion = self.__campos(self.frame_prediciones, "Entrada", self.cantidad_entradas.get())

        if len(str(self.resultado_prediccion_perceptron)) > 0:
            label3 = ttk.Label(self.frame_prediciones, text="El resultado es = {}".format(self.resultado_prediccion_perceptron))
            label3.grid(column=0, row=5, padx=4, pady=7)

    def __calcular_prediccion(self):
        self.frame_prediciones.grid_remove()
        self.entradas_prediccion = self.__cargar_arreglos(self.entradas_prediccion, self.cantidad_entradas.get(), default_cero=True)
        self.resultado_prediccion_perceptron = self.red_perceptron.prediccion(self.entradas_prediccion)
        del self.entradas_prediccion[:]
        self.__frame_prediciones()

    def __frame_boton_accion_grafica(self):
        frame_boton_accion = ttk.LabelFrame(self.labelframe_grafica, text="Acción")
        frame_boton_accion.grid(column=0, row=3, ipadx=270, sticky=NW)

        boton_calcular = ttk.Button(frame_boton_accion, text='Predeccir', state=NORMAL,
                                   command=lambda: self.__calcular_prediccion())
        boton_calcular.pack(side=TOP, padx=16, pady=15)

        boton_inicion = ttk.Button(frame_boton_accion, text='Volver Atrás', state=NORMAL,
                                   command=lambda: self.__frame_parametros_perceptron())
        boton_inicion.pack(side=TOP, padx=16, pady=15)


    # ---------------------------------------------------------------------------#


    def __frame_parametros_rns(self):
        try:
            del self.entradas_prediccion_rns[:]
            self.labelframe_grafica_rns.grid_remove()
        except:
            pass
        self.labelframe_parametros_rns = ttk.LabelFrame(self.pagina2, text="Parametros Red Neuronal Simple:")
        self.labelframe_parametros_rns.grid(column=0, row=0, ipady=1880)
        self.__frame_entradas_rns()
        self.__frame_entrenamiento_rns()
        self.__frame_boton_accion_rns()

    def __frame_entradas_rns(self):
        frame_entradas_rns = ttk.LabelFrame(self.labelframe_parametros_rns, text="Cantidad Entradas:")
        frame_entradas_rns.grid(column=0, row=0, ipadx=95, pady=10, padx=0, sticky=NW)
        label_cant_entradas = ttk.Label(frame_entradas_rns,
                                        text="Seleccione la cantidad de entradas que tendra la Red Neuronal:")
        label_cant_entradas.grid(column=0, row=0, padx=4, pady=4)
        ttk.Radiobutton(frame_entradas_rns, text='2', variable=self.cantidad_entradas_rns, value=2,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=1, row=1)
        ttk.Radiobutton(frame_entradas_rns, text='3', variable=self.cantidad_entradas_rns, value=3,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=2, row=1)
        ttk.Radiobutton(frame_entradas_rns, text='4', variable=self.cantidad_entradas_rns, value=4,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=3, row=1)
        ttk.Radiobutton(frame_entradas_rns, text='5', variable=self.cantidad_entradas_rns, value=5,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=4, row=1)

    def __cambio_cant_datos_entren_rns(self):
        del self.entradas_set_rns[:]
        del self.salidas_set_rns[:]
        self.frame_entrenamiento_rns.destroy()
        self.__frame_entrenamiento_rns()

    def __frame_entrenamiento_rns(self):
        self.frame_entrenamiento_rns = ttk.LabelFrame(self.labelframe_parametros_rns, text="Set de entrenamiento Red Neuronal Simple")
        self.frame_entrenamiento_rns.grid(column=0, row=4, ipadx=10, sticky=NW)

        label_cant_epocas = ttk.Label(self.frame_entrenamiento_rns, text="Seleccione la cantidad epocas (repeticiones):")
        label_cant_epocas.grid(column=0, row=0, padx=4, pady=4)
        self.epocas_rns = ttk.Entry(self.frame_entrenamiento_rns)
        self.epocas_rns.grid(column=0, row=1, padx=4, pady=4)

        label_cant_datos_entre = ttk.Label(self.frame_entrenamiento_rns, text="Seleccione la cantidad de datos de entrenamiento:")
        label_cant_datos_entre.grid(column=0, row=2, padx=4, pady=4)
        ttk.Radiobutton(self.frame_entrenamiento_rns, text='4', variable=self.cantidad_datos_entrenamiento_rns, value=4,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=0, row=3)
        ttk.Radiobutton(self.frame_entrenamiento_rns, text='10', variable=self.cantidad_datos_entrenamiento_rns, value=10,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=0, row=4)
        ttk.Radiobutton(self.frame_entrenamiento_rns, text='15', variable=self.cantidad_datos_entrenamiento_rns, value=15,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=0, row=5)
        ttk.Radiobutton(self.frame_entrenamiento_rns, text='20', variable=self.cantidad_datos_entrenamiento_rns, value=20,
                        command=lambda: self.__cambio_cant_datos_entren_rns()).grid(column=0, row=6)

        label_entrenameinto = ttk.Label(self.frame_entrenamiento_rns,
                                        text="Ingrese las entradas y salidas esperadas para el entrenamiento:",
                                        justify=LEFT)
        label_entrenameinto.grid(column=0, row=7, pady=7)

        self.__set_entrenamiento_rns()

    def __set_entrenamiento_rns(self):
        cantidad_datos = self.cantidad_datos_entrenamiento_rns.get()
        cantidad_entradas = self.cantidad_entradas_rns.get()
        row = 7  # indica la fila desde donde inicia
        self.salidas_set_rns = []
        self.entradas_set_rns = []
        for i in range(cantidad_datos):
            row += i
            subframe_entrenamiento_rns = ttk.LabelFrame(self.frame_entrenamiento_rns, text="Set " + str(i + 1))
            subframe_entrenamiento_rns.grid(column=0, row=row, ipadx=100, padx=30, pady=7, sticky=NW)
            lista_entradas = self.__campos_entrenamiento(subframe_entrenamiento_rns, self.salidas_set_rns, row, cantidad_entradas)
            self.entradas_set_rns.append(lista_entradas)
            del lista_entradas[:cantidad_datos:-1]

    def __frame_boton_accion_rns(self):
        frame_boton_accion = ttk.LabelFrame(self.labelframe_parametros_rns, text="Acción")
        frame_boton_accion.grid(column=0, row=5, ipadx=270, sticky=NW)

        boton_inicion = ttk.Button(frame_boton_accion, text='Calcular', state=NORMAL,
                                   command=lambda: self.__crear_red_neuronal_simple())
        boton_inicion.pack(side=TOP, padx=16, pady=15)

    def __crear_red_neuronal_simple(self):
        valida_campo = lambda x: float(x) if len(x) > 0 else 0
        cantidad_entradas = int(self.cantidad_entradas_rns.get())
        epocas = int(valida_campo(self.epocas_rns.get()))

        entradas_entrenamiento = list()
        for lista in self.entradas_set_rns:
            result = self.__cargar_arreglos(lista, self.cantidad_datos_entrenamiento_rns.get())
            entradas_entrenamiento.append(result)
        
        salidas_esperadas = self.__cargar_arreglos(self.salidas_set_rns, self.cantidad_datos_entrenamiento_rns.get())

        self.red_neuronal_simple.run(cantidad_entradas=cantidad_entradas, numero_iteraciones=epocas,
                                entradas=entradas_entrenamiento, salidas=salidas_esperadas)

        # se destruye el frame que contine el formulario de parametros de la RNS
        # se crea frame que contine formuario para realizar predicción y gráfica
        self.labelframe_parametros_rns.grid_forget()
        self.__frame_grafica_rns()

    def __frame_grafica_rns(self):
        self.labelframe_grafica_rns = ttk.LabelFrame(self.pagina2, text="Red Neuronal Simple:")
        self.labelframe_grafica_rns.grid(column=0, row=0)
        self.__frame_rns_calculada()
        self.__frame_plot_rns()
        self.__frame_prediciones_rns()
        self.__frame_boton_accion_grafica_rns()

    def __frame_rns_calculada(self):
        frame_rns_calculada = ttk.LabelFrame(self.labelframe_grafica_rns, text="Pesos calculados")
        frame_rns_calculada.grid(column=0, row=0, ipadx=214, sticky=NW)
        
        pesos_calculados = self.red_neuronal_simple.obtener_pesos() 
        texto = "Pesos calculados por la Red Neuronal Simple:\n\n"
        for i,peso in enumerate(pesos_calculados):
            texto += ("peso {} = {} \n".format(i, round(peso[0], 5)))
        label = ttk.Label(frame_rns_calculada, text=texto)
        label.grid(column=0, row=0, padx=5, pady=10)

    def __frame_plot_rns(self):
        frame_plot = ttk.LabelFrame(self.labelframe_grafica_rns, text="Gráfica Aprendizaje Red Neuronal Simple")
        frame_plot.grid(column=0, row=1, ipadx=48, sticky=NW)
        errores = self.red_neuronal_simple.obtener_errores()

        fig = Figure(figsize=(5.5,4.5))
        a = fig.add_subplot(111)
        
        for i in range(int(self.cantidad_entradas_rns.get())):
            label = "Peso entrada {}".format(i+1)
            conexion = list()
            for error in errores:
                conexion.append(error[i])
            a.plot(conexion, label = label)
            conexion = []
        a.set_title ("Aprendizaje de la Red Neuronal ", fontsize=10)
        a.set_ylabel("Valores Error", fontsize=10)
        a.set_xlabel("Epocas", fontsize=10)
        a.grid()
        a.legend()
        self.canvas2 = FigureCanvasTkAgg(fig, master=frame_plot)
        self.canvas2.get_tk_widget().pack()
        self.canvas2.draw()

    def __frame_prediciones_rns(self):
        self.frame_prediciones_rns = ttk.LabelFrame(self.labelframe_grafica_rns, text="Predicciones")
        self.frame_prediciones_rns.grid(column=0, row=2, ipadx=228, sticky=NW)

        self.entradas_prediccion_rns = self.__campos(self.frame_prediciones_rns, "Entrada", self.cantidad_entradas_rns.get())
        if len(str(self.resultado_prediccion_rns)) > 0:
            label3 = ttk.Label(self.frame_prediciones_rns, text="El resultado es = {}".format(self.resultado_prediccion_rns))
            label3.grid(column=0, row=5, padx=4, pady=7)

    def __calcular_prediccion_rns(self):
        self.frame_prediciones_rns.grid_remove()
        self.entradas_prediccion_rns = self.__cargar_arreglos(self.entradas_prediccion_rns, self.cantidad_entradas_rns.get(), default_cero=True)
        self.resultado_prediccion_rns = self.red_neuronal_simple.prediccion(self.entradas_prediccion_rns)
        del self.entradas_prediccion_rns[:]
        self.__frame_prediciones_rns()

    def __frame_boton_accion_grafica_rns(self):
        frame_boton_accion = ttk.LabelFrame(self.labelframe_grafica_rns, text="Acción")
        frame_boton_accion.grid(column=0, row=3, ipadx=270, sticky=NW)

        boton_calcular = ttk.Button(frame_boton_accion, text='Predeccir', state=NORMAL,
                                   command=lambda: self.__calcular_prediccion_rns())
        boton_calcular.pack(side=TOP, padx=16, pady=15)

        boton_inicion = ttk.Button(frame_boton_accion, text='Volver Atrás', state=NORMAL,
                                   command=lambda: self.__frame_parametros_rns())
        boton_inicion.pack(side=TOP, padx=16, pady=15)

#------------------------------------------------------------------------#

    def __frame_parametros_rnm(self):
        try:
            self.resultado_prediccion_rnm = ""
            self.labelframe_grafica_rnm.grid_remove()
        except:
            pass
        self.labelframe_parametros_rnm = ttk.LabelFrame(self.pagina3, text="Parametros Perceptrón:")
        self.labelframe_parametros_rnm.grid(column=0, row=0, ipady=1880)
        self.__frame_capas_rnm()
        self.__frame_fun_activacion()
        self.__frame_entrenamiento_rnm()
        self.__frame_boton_accion_rnm()

    def __frame_capas_rnm(self):
        frame_capas = ttk.LabelFrame(self.labelframe_parametros_rnm, text="Capas:")
        frame_capas.grid(column=0, row=0, ipadx=170, pady=10, padx=0, sticky=NW)
        values = [1,2,3,4]
        label_capa_entrada = ttk.Label(frame_capas,
                                        text="Seleccione la cantidad de neuronas en la capa de entrada:")
        label_capa_entrada.grid(column=0, row=0, pady = 10)
        self.capa_entrada = ttk.Combobox(frame_capas, state="readonly", values=values)
        self.capa_entrada.set(2)
        self.capa_entrada.bind("<<ComboboxSelected>>", self.__cambio_cantidades_rnm)
        self.capa_entrada.grid(column=0, row=1)

        label_capa_oculta = ttk.Label(frame_capas,
                                        text="Seleccione la cantidad de neuronas en la capa oculta:")
        label_capa_oculta.grid(column=0, row=2, pady = 10)
        self.capa_oculta = ttk.Combobox(frame_capas, state="readonly", values=values)
        self.capa_oculta.set(3)
        self.capa_oculta.bind("<<ComboboxSelected>>", self.__cambio_cantidades_rnm)
        self.capa_oculta.grid(column=0, row=3)

        label_capa_salida = ttk.Label(frame_capas,
                                        text="Seleccione la cantidad de neuronas en la capa oculta:")
        label_capa_salida.grid(column=0, row=4, pady = 10)
        self.capa_salida = ttk.Combobox(frame_capas, state="readonly", values=values)
        self.capa_salida.set(2)
        self.capa_salida.bind("<<ComboboxSelected>>", self.__cambio_cantidades_rnm)
        self.capa_salida.grid(column=0, row=5)

    def __cambio_cantidades_rnm(self, event):
        self.frame_sets.destroy()
        self.__set_entrenamiento_rnm()

    def __frame_fun_activacion(self):
        frame_fun_activacion = ttk.LabelFrame(self.labelframe_parametros_rnm, text="Función de activación:")
        frame_fun_activacion.grid(column=0, row=1, ipadx=170, pady=10, padx=0, sticky=NW)
        values = ["sigmoide", "tangente"]
        label_funcion_activacion = ttk.Label(frame_fun_activacion,
                                        text="Seleccione la función de activacion para la Red Multicapa:")
        label_funcion_activacion.grid(column=0, row=0, pady = 10)
        self.funcion_activacion = ttk.Combobox(frame_fun_activacion, state="readonly", values=values)
        self.funcion_activacion.current(1)
        self.funcion_activacion.grid(column=0, row=1)

    def __frame_entrenamiento_rnm(self):
        self.frame_entrenamiento_rnm = ttk.LabelFrame(self.labelframe_parametros_rnm, text="Set de entrenamiento Red Neuronal Multicapa")
        self.frame_entrenamiento_rnm.grid(column=0, row=2, ipadx=87, sticky=NW)

        label_cant_epocas = ttk.Label(self.frame_entrenamiento_rnm, text="Seleccione la cantidad epocas:")
        label_cant_epocas.grid(column=0, row=0, padx=4, pady=4)
        self.epocas_rnm = ttk.Entry(self.frame_entrenamiento_rnm)
        self.epocas_rnm.grid(column=0, row=1, padx=4, pady=4)

        label_fact_aprendizaje = ttk.Label(self.frame_entrenamiento_rnm, text="Seleccione el factor de Aprendizaje:")
        label_fact_aprendizaje.grid(column=0, row=2, padx=4, pady=4)
        self.fact_aprendizaje = ttk.Entry(self.frame_entrenamiento_rnm)
        self.fact_aprendizaje.grid(column=0, row=3, padx=4, pady=4)

        label_cant_datos_entre = ttk.Label(self.frame_entrenamiento_rnm, text="Seleccione la cantidad de datos de entrenamiento:")
        label_cant_datos_entre.grid(column=0, row=4, padx=4, pady=5)
        values = [7,8,12,16]
        self.cantidad_datos_entrenamiento_rnm = ttk.Combobox(self.frame_entrenamiento_rnm, state="readonly", values=values)
        self.cantidad_datos_entrenamiento_rnm.set(7)
        self.cantidad_datos_entrenamiento_rnm.bind("<<ComboboxSelected>>", self.__cambio_cantidades_rnm)
        self.cantidad_datos_entrenamiento_rnm.grid(column=0, row=5)

        label_entrenameinto = ttk.Label(self.frame_entrenamiento_rnm,
                                        text="Ingrese las entradas y salidas esperadas para el entrenamiento:",
                                        justify=LEFT)
        label_entrenameinto.grid(column=0, row=6, pady=7)


        self.__set_entrenamiento_rnm()

    def __set_entrenamiento_rnm(self):
        self.frame_sets = ttk.Frame(self.frame_entrenamiento_rnm)
        self.frame_sets.grid(column = 0, row = 7)
        cantidad_datos = int(self.cantidad_datos_entrenamiento_rnm.get())
        cantidad_entradas = self.capa_entrada.get()
        cantidad_salidas = self.capa_salida.get()
        row = 0  # indica la fila desde donde inicia
        self.salidas_set_rnm = []
        self.entradas_set_rnm = []
        for i in range(cantidad_datos):
            row += i
            subframe_entrenamiento_rnm = ttk.LabelFrame(self.frame_sets, text="Set " + str(i + 1))
            subframe_entrenamiento_rnm.grid(column=0, row=row, ipadx=30, padx=5, pady=6, sticky=NW)
            lista_entradas = self.__campos_entrenamiento(subframe_entrenamiento_rnm, self.salidas_set_rnm, row, cantidad_entradas,cantidad_salidas=cantidad_salidas)
            self.entradas_set_rnm.append(lista_entradas)
            del lista_entradas[:cantidad_datos:-1]

    def __frame_boton_accion_rnm(self):
        frame_boton_accion = ttk.LabelFrame(self.labelframe_parametros_rnm, text="Acción")
        frame_boton_accion.grid(column=0, row=7, ipadx=270, sticky=NW)

        boton_inicion = ttk.Button(frame_boton_accion, text='Calcular', state=NORMAL,
                                   command=lambda: self.__crear_red_neuronal_multicapa())
        boton_inicion.pack(side=TOP, padx=16, pady=15)
    
    def __crear_red_neuronal_multicapa(self):
        valida_campo = lambda x: float(x) if len(x) > 0 else 0.0
        capas = [int(self.capa_entrada.get()), int(self.capa_oculta.get()), int(self.capa_salida.get())]
        activacion = self.funcion_activacion.get()
        factor_aprendizaje = valida_campo(self.fact_aprendizaje.get())
        epocas = int(valida_campo(self.epocas_rnm.get()))

        entradas_entrenamiento = list()
        for lista in self.entradas_set_rnm:
            result = self.__cargar_arreglos(lista, self.cantidad_datos_entrenamiento_rnm.get())
            entradas_entrenamiento.append(result)
        entradas_entrenamiento = array(entradas_entrenamiento
            )
        salidas_esperadas = list()
        for lista_salida in self.salidas_set_rnm:
            result_salidas = self.__cargar_arreglos(lista_salida, self.cantidad_datos_entrenamiento_rnm.get())
            salidas_esperadas.append(result_salidas)
        salidas_esperadas = array(salidas_esperadas)

        self.red_neuronal_multicapa.run(capas=capas, activacion = activacion)
        self.red_neuronal_multicapa.ajuste( X = entradas_entrenamiento, y = salidas_esperadas, factor_aprendizaje = factor_aprendizaje, epocas = epocas)

        # se destruye el frame que contine el formulario de parametros de la RNM 
        # se crea frame que contine formuario para realizar predicción y gráfica
        self.labelframe_parametros_rnm.grid_forget()
        self.__frame_grafica_rnm()

    def __frame_grafica_rnm(self):
        self.labelframe_grafica_rnm = ttk.LabelFrame(self.pagina3, text="Red Neuronal Multicapa:")
        self.labelframe_grafica_rnm.grid(column=0, row=0, pady = 10)
        self.__frame_rnm_calculada()
        self.__frame_plot_rnm()
        self.__frame_prediciones_rnm()
        self.__frame_boton_accion_grafica_rnm()

    def __frame_rnm_calculada(self):
        frame_rnm_calculada = ttk.LabelFrame(self.labelframe_grafica_rnm, text="Pesos calculados")
        frame_rnm_calculada.grid(column=0, row=0, ipadx=123, sticky=NW)
        
        pesos_calculados = self.red_neuronal_multicapa.obtener_pesos() 
        texto = "Pesos calculados por la Red Neuronal Simple:\n\n"
        texto = ("Pesos capa entrada  = {} \n".format(pesos_calculados[0]))
        label = ttk.Label(frame_rnm_calculada, text=texto)
        label.grid(column=0, row=0, padx=5, pady=10)

        texto2 = ("Pesos capa salida  = {} \n".format(pesos_calculados[1]))
        label2 = ttk.Label(frame_rnm_calculada, text=texto2)
        label2.grid(column=0, row=1, padx=5, pady=10)

    def __frame_plot_rnm(self):
        frame_plot = ttk.LabelFrame(self.labelframe_grafica_rnm, text="Gráfica Aprendizaje Red Neuronal Simple")
        frame_plot.grid(column=0, row=1, ipadx=48, sticky=NW)
        deltas = self.red_neuronal_multicapa.obtener_deltas()
        fig = Figure(figsize=(5.5,4.5))
        a = fig.add_subplot(111)
        valores = []
        for arreglo in deltas:
            valores.append(arreglo[1][0] + arreglo[1][1])
        
        a.plot(range(len(valores)), valores, color="blue")
        a.set_title ("Aprendizaje de la Red Neuronal ", fontsize=10)
        a.set_ylabel("Costo", fontsize=10)
        a.set_xlabel("Epocas", fontsize=10)
        a.grid()
        self.canvas3 = FigureCanvasTkAgg(fig, master=frame_plot)
        self.canvas3.get_tk_widget().pack()
        self.canvas3.draw()

    def __frame_prediciones_rnm(self):
        self.frame_prediciones_rnm = ttk.LabelFrame(self.labelframe_grafica_rnm, text="Predicciones")
        self.frame_prediciones_rnm.grid(column=0, row=2, ipadx=228, sticky=NW)

        self.entradas_prediccion_rnm = self.__campos(self.frame_prediciones_rnm, "Entrada", self.capa_entrada.get())

        if len(str(self.resultado_prediccion_rnm)) > 0:
            label = ttk.Label(self.frame_prediciones_rnm, text="El resultado es = {}".format(self.resultado_prediccion_rnm))
            label.grid(column=0, row=5, padx=4, pady=7)

    def __calcular_prediccion_rnm(self):
        self.entradas_prediccion_rnm = self.__cargar_arreglos(self.entradas_prediccion_rnm, self.capa_entrada.get(), default_cero=True)
        self.entradas_prediccion_rnm = array(self.entradas_prediccion_rnm)
        self.resultado_prediccion_rnm = self.red_neuronal_multicapa.predecir(self.entradas_prediccion_rnm)
        
        self.frame_prediciones_rnm.grid_remove()
        self.__frame_prediciones_rnm()

    def __frame_boton_accion_grafica_rnm(self):
        frame_boton_accion = ttk.LabelFrame(self.labelframe_grafica_rnm, text="Acción")
        frame_boton_accion.grid(column=0, row=3, ipadx=270, sticky=NW)

        boton_calcular = ttk.Button(frame_boton_accion, text='Predeccir', state=NORMAL,
                                   command=lambda: self.__calcular_prediccion_rnm())
        boton_calcular.pack(side=TOP, padx=16, pady=15)

        boton_inicion = ttk.Button(frame_boton_accion, text='Volver Atrás', state=NORMAL,
                                   command=lambda: self.__frame_parametros_rnm())
        boton_inicion.pack(side=TOP, padx=16, pady=15)

def main():
    red_perceptron = perceptron.Perceptron()
    red_neuronal_simple = redneuronalsimple.RedNeuronalSimple()
    red_neuronal_multicapa = redneuronalmulticapa.RedNeuronalMulticapa()
    app = Graficadora(red_perceptron, red_neuronal_simple, red_neuronal_multicapa)


if __name__ == "__main__":
    main()