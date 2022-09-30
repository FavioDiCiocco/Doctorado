#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:08:53 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import funciones as func
import deffuant as deff


##################################################################################
##################################################################################

# CÓDIGO PRINCIPAL

##################################################################################
##################################################################################

# Empiezo a medir el tiempo para ver cuánto tarda el programa
t0 = time.time()


# Definición de los parámetros del modelo
#---------------------------------------------------------------------------

N = 30 # Número de agentes
mu = 0.25 # Parámetro de convergencia
epsilon = 0.6 # Tolerancia de opiniones

Opiniones = deff.Distribucion_inicial(N) # Armo la distribución inicial de opiniones


# Evolución del sistema
#----------------------------------------------------------------------------

Opiniones_previas = np.array([opi for opi in Opiniones]) # Tengo que definir Opiniones_previas
# así para que me los considere arrays separados, sino Opiniones_previas apunta al mismo
# espacio de memoria.

for iteracion in range(N):
    # Elijo los dos agentes que van a interactuar
    Agentes = rng.choice(np.arange(N),size=2,replace=False)
    # Los agentes interactúan si sus opiniones se encuentran suficientemente cerca
    distancia = Opiniones[Agentes[1]]-Opiniones[Agentes[0]] # Distancia entre las opiniones de los agentes
    if np.abs(distancia) < epsilon:
        # Evoluciono al primer agente
        Opiniones[Agentes[0]] = Opiniones[Agentes[0]] + mu*(distancia)
        # Evoluciono al segundo agente
        Opiniones[Agentes[1]] = Opiniones[Agentes[1]] - mu*(distancia)
        # El segundo tiene un menos porque quiero -diferencia, cosa de 
        # que el segundo agente se acerque al primero

# Parámetro de corte
#---------------------------------------------------------------------------

# El sistema debe llegar a un final en el que los agentes forman grupos, uno o varios.
# Propongo usar la diferencia entre los vectores opinión para determinar
# si el sistema llega a un estado estable
desviacion_estandar = np.std(np.abs(Opiniones-Opiniones_previas))

# Gráfico del sistema
#----------------------------------------------------------------------------

plt.rcParams.update({'font.size': 24})
fig,ax = plt.subplots(figsize=(12,8)) # Creo la figura que voy a graficar

T = np.ones(N)*t # Largo de mi grilla
# Los valores de Y serán un array que contenga las opiniones de todos los agentes
# en cada t. Es más, podría directamente graficar eso usando el array de Opiniones
# Y no tener un array que crezca infinitamente.

# Grafico
ax.plot(T,Opiniones,"xg",markersize=4)
plt.savefig( path+"/Deffuant_mu={}_eps={}.png".format(mu,epsilon), bbox_inches = "tight")
plt.close()

