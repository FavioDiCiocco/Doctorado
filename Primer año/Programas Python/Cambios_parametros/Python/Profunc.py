#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()


"""
###################################################################################################

# Defino la cantidad de agentes
N = 1000

# Cargo el archivo con la matriz de adyacencia Random Regulars

Datos = func.ldata("../../../Programas C/MARE/Random_Regulars/Random-regular_N=1000_ID=1.file")
Adyacencia  = np.reshape(np.array([i for i in Datos[0][:-1:]],dtype = "int"),(N,N))

# Armo el grafo a partir de la matriz de Adyacencia

G = nx.from_numpy_matrix(Adyacencia)

# Dada la matriz de adyacencia, ahora hago el catalogar a los agentes según su distancia
# al primer agente. Uso un diccionario para catalogarlo bien.

distancia = 1 # Esto lo uso para marcar la distancia entre agentes
Registrados = set([0]) # Estos son los agentes que revisé de la red
Agentes_catalogados = dict() # Diccionario de agentes según su distancia al primer nodo
Agentes_catalogados[0] = set([0]) # Fijo al primer agente a distancia cero de sí mismo


while len(Registrados) != N :
    Vecinos = [] # En esta lista anoto todos los agentes visitados
    Descarte = [] # En esta lista pongo los agentes que voy a descartar de Conjunto_vecinos
    for agente in Agentes_catalogados[distancia-1]:
        for vecino in G.neighbors(agente):
            Vecinos.append(vecino) # Me apendeo los vecinos de "agente"
    Conjunto_vecinos = set(Vecinos) # Elimino los duplicados
    for agente in Conjunto_vecinos:
        if agente in Registrados:
            Descarte.append(agente) # Anoto los elementos previamente registrados
        Registrados.add(agente) # Registro todos los agentes visitados
    for agente in Descarte:
        Conjunto_vecinos.discard(agente) # Descarto los elementos previamente registrados
    # Me anoto el conjunto de agentes que se encuentran a distancia "distancia"
    Agentes_catalogados[distancia] = Conjunto_vecinos
    distancia +=1 # Paso a mirar a los agentes en la siguiente distancia

####################################################################################################

# Ya estudié la matriz de adyacencia, ahora debería revisar si mi función de catalogación funciona mejor

Datos = func.ldata("../categorizacion_prueba.file")
Categorias = np.array(Datos[2][:-1],dtype = "int")
"""

####################################################################################################
####################################################################################################
####################################################################################################

# Acá voy a hacer lo de analizar las curvas de Kappa en función de epsilon y alfa. Para eso primero
# tengo que a mano identificar la curva de la transición.

Epsilon = np.arange(0.5,3.1,0.1) # Estos son los valores de Epsilon graficados

#------------------------------------------------------------------------------------------------------------

# Defino a ojo los valores de Kappa asociados y los voy anotando uno a uno

# Alfa = 2

Datos_Alfa_2 = np.zeros(Epsilon.shape)
Datos_Alfa_2[0] = 0.6
Datos_Alfa_2[1] = 0.7
Datos_Alfa_2[2] = 0.7
Datos_Alfa_2[3] = 0.8
Datos_Alfa_2[4] = 0.8
Datos_Alfa_2[5] = 0.9
Datos_Alfa_2[6] = 1
Datos_Alfa_2[7] = 1.1
Datos_Alfa_2[8] = 1.1
Datos_Alfa_2[9] = 1.2
Datos_Alfa_2[10] = 1.3
Datos_Alfa_2[11] = 1.4
Datos_Alfa_2[12] = 1.5
Datos_Alfa_2[13] = 1.6
Datos_Alfa_2[14] = 1.75
Datos_Alfa_2[15] = 2
Datos_Alfa_2[16] = 2.1
Datos_Alfa_2[17] = 2.2
Datos_Alfa_2[18] = 2.4
Datos_Alfa_2[19] = 2.6
Datos_Alfa_2[20] = 2.8
Datos_Alfa_2[21] = 3
Datos_Alfa_2[22] = 3.2
Datos_Alfa_2[23] = 3.5
Datos_Alfa_2[24] = 3.8
Datos_Alfa_2[25] = 4.2

#------------------------------------------------------------------------------------------------------------

# Alfa = 4

Datos_Alfa_4 = np.zeros(Epsilon.shape)
Datos_Alfa_4[0] = 0.5
Datos_Alfa_4[1] = 0.5
Datos_Alfa_4[2] = 0.6
Datos_Alfa_4[3] = 0.6
Datos_Alfa_4[4] = 0.6
Datos_Alfa_4[5] = 0.6
Datos_Alfa_4[6] = 0.6
Datos_Alfa_4[7] = 0.6
Datos_Alfa_4[8] = 0.7
Datos_Alfa_4[9] = 0.7
Datos_Alfa_4[10] = 0.8
Datos_Alfa_4[11] = 0.8
Datos_Alfa_4[12] = 0.9
Datos_Alfa_4[13] = 0.9
Datos_Alfa_4[14] = 1
Datos_Alfa_4[15] = 1
Datos_Alfa_4[16] = 1
Datos_Alfa_4[17] = 1.1
Datos_Alfa_4[18] = 1.2
Datos_Alfa_4[19] = 1.3
Datos_Alfa_4[20] = 1.4
Datos_Alfa_4[21] = 1.5
Datos_Alfa_4[22] = 1.6
Datos_Alfa_4[23] = 1.7
Datos_Alfa_4[24] = 1.8
Datos_Alfa_4[25] = 2.1


#------------------------------------------------------------------------------------------------------------

# Alfa = 6

Datos_Alfa_6 = np.zeros(Epsilon.shape)
Datos_Alfa_6[0] = 0.5
Datos_Alfa_6[1] = 0.5
Datos_Alfa_6[2] = 0.5
Datos_Alfa_6[3] = 0.5
Datos_Alfa_6[4] = 0.5
Datos_Alfa_6[5] = 0.5
Datos_Alfa_6[6] = 0.5
Datos_Alfa_6[7] = 0.5
Datos_Alfa_6[8] = 0.5
Datos_Alfa_6[9] = 0.5
Datos_Alfa_6[10] = 0.5
Datos_Alfa_6[11] = 0.5
Datos_Alfa_6[12] = 0.6
Datos_Alfa_6[13] = 0.6
Datos_Alfa_6[14] = 0.6
Datos_Alfa_6[15] = 0.6
Datos_Alfa_6[16] = 0.7
Datos_Alfa_6[17] = 0.7
Datos_Alfa_6[18] = 0.8
Datos_Alfa_6[19] = 0.8
Datos_Alfa_6[20] = 0.9
Datos_Alfa_6[21] = 1
Datos_Alfa_6[22] = 1
Datos_Alfa_6[23] = 1.1
Datos_Alfa_6[24] = 1.2
Datos_Alfa_6[25] = 1.4

#------------------------------------------------------------------------------------------------------------

# Teniendo los datos, ahora relizo el ajuste para obtener el exponente asociado al alfa
# Supongo que kappa sigue la siguiente relación: K = \frac{epsilon}{alpha^r}+C

Resultados = dict()

for indice,Datos in enumerate([Datos_Alfa_2,Datos_Alfa_4,Datos_Alfa_6]):
    
    alfa = (indice+1)*2 # Defino alfa en base al indice
    
    # Defino la función con la que voy a hacer el ajuste

    def Kappa(X,C):
        return C*(1+math.e**(-alfa*C+X))
    
    # Calculo los parámetros
    
    parametros_optimos,parametros_covarianza = curve_fit(Kappa,Epsilon,Datos)
    
    # Calculo el error
    
    error_C = math.sqrt(parametros_covarianza[0,0])
    
    print(r"El valor de C asociado al alfa={} es: {} $\pm$ {}".format(alfa, parametros_optimos[0],error_C))
    # print(r"El valor de C asociado al alfa={} es: {} $\pm$ {}".format(alfa, parametros_optimos[1],error_C))
    # print(r"El valor de D asociado al alfa={} es: {} $\pm$ {}".format(alfa, parametros_optimos[2],error_D))
    
    # Guardo los valores en mi diccionario
    
    Resultados[alfa] = parametros_optimos
 

#------------------------------------------------------------------------------------------------------------
    
# Ploteo las curvas reales contra las curvas ajustadas

plt.rcParams.update({'font.size': 32})
plt.figure("Curva_transicion",figsize=(20,15))
colores = ["b","g","r"]
for indice,Y in enumerate([Datos_Alfa_2,Datos_Alfa_4,Datos_Alfa_6]):
    
    alfa = (indice+1)*2 # Defino alfa en base al indice
        
    plt.plot(Epsilon,Y,color = colores[indice], marker = "*", linestyle = "None" , markersize = 12)
    
    # Defino la función con la que voy a hacer el ajuste

    def Kappa(X,C):
        return C*(1+math.e**(-alfa*C+X))
    
    Y_calculado = Kappa(Epsilon,Resultados[alfa][0])
    plt.plot(Epsilon, Y_calculado, label = r"$\alpha =$ {}".format(alfa),color = colores[indice], linewidth=4)
    
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\kappa$")
plt.legend()
plt.grid(alpha = 0.5)
plt.show()





func.Tiempo(t0)
