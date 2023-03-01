#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
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

# Hago pruebas para hallar soluciones a funciones

alfa = 0.5
epsilon = 3

def Derivada_kappa(x,alfa,epsilon):
    return np.exp(alfa*x-epsilon)+1-alfa*x

roots_min = fsolve(Derivada_kappa,0,args=(alfa,epsilon))
print(roots_min[0])
roots_max = fsolve(Derivada_kappa,8,args=(alfa,epsilon))
print(roots_max[0])

def Kappa(x,alfa,epsilon):
    return x*( 1 + np.exp(-alfa*x +epsilon) )

print("El kappa_max es: ",Kappa(roots_min[0],alfa,epsilon))
print("El kappa_min es: ",Kappa(roots_max[0],alfa,epsilon))


func.Tiempo(t0)
