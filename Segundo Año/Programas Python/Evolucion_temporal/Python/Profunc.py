#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()



###################################################################################################

# Defino la cantidad de agentes
N = 1000

# Cargo el archivo con la matriz de adyacencia Erdos-Renyi

Datos = func.ldata("../MARE_Algarve/Erdos-Renyi/ErdosRenyi_N=1000_ID=51.file")
Adyacencia  = np.reshape(np.array([i for i in Datos[0][:-1:]],dtype = "int"),(N,N))

# Armo el grafo a partir de la matriz de Adyacencia

G = nx.from_numpy_matrix(Adyacencia)

# gradomedio = 0
# for nodo in G.nodes():
#     gradomedio += G.degree[nodo]

# gradomedio /= N

# print("El grado medio de la red es: ", gradomedio)

Vecinos = [nodo for nodo in G.neighbors(79)]
print(Vecinos)
