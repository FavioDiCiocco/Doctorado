#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:40:50 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import funciones as func
import votante as vote


##################################################################################
##################################################################################

# CÓDIGO PRINCIPAL

##################################################################################
##################################################################################

# Empiezo a medir el tiempo para ver cuánto tarda el programa
t0 = time.time()


# Definición de los parámetros del modelo
#---------------------------------------------------------------

L = 50 # Ancho y largo de la grilla
p = 0.5 # Distribución de los agentes positivos y los negativos


G = vote.Construccion_grilla_cuadrada(L,p) # Construyo la red y asigno las posturas
vote.Graficar_y_guardar_sistema(G,L,0) # Me guardo un gráfico del sistema.
# Se le puede asignar un path para que guarde el archivo en la carpeta que quiera



# Medición del parámetro de corte
#-------------------------------------------------------------------------------

# activos = [nx.get_node_attributes(G, "Postura")[i]*nx.get_node_attributes(G, "Postura")[j] for i,j in G.edges()].count(-1)

# fraccion_activos = activos/len(G.edges())

# print("Calculé la fracción de agentes activos")
# func.Tiempo(t0)

# Visualización del sistema durante su evolución
#------------------------------------------------------------------------------
# Primero creo la figura sobre la que voy a continuamente graficar mi sistema
fig,ax = plt.subplots(figsize=(12,8))

# Por lo que vi, puedo graficar el pcolormesh usando arrays, no necesito 
# necesariamente que X e Y sean grids también.

# Por lo que vi x e y tienen que tener un tamaño mayor en 1 en cada respectiva
# dimensión comparados con el grid de Z
x = np.arange(L)
y = np.arange(L)

# Me armo mi grid de posturas de los agentes
Z = np.zeros((L,L))

# Relleno el grid haciendo uso de que los nodos se identifican por su posición
# en la grilla

for objeto in nx.get_node_attributes(G,"Postura").items():
    Z[objeto[0][0],objeto[0][1]] = objeto[1]

ax.pcolormesh(x,y,Z,shading="nearest",cmap="bwr")
ax.set_title("Gráfico final")
plt.show()

print("Armé el gráfico de agentes activos")
func.Tiempo(t0)

