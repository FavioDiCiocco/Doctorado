#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:36:18 2022

@author: favio
"""

# Importo los paquetes necesarios para las funciones

import matplotlib.pyplot as plt
import numpy as np

##################################################################################
##################################################################################

# FUNCIONES

##################################################################################
##################################################################################

# Armo una distribución aleatoria de opiniones entre 0 y 1 para todos los agentes

def Distribucion_inicial(N):
    rng = np.random.default_rng() # Objeto de numpy que genera distribuciones números aleatorios y sampleos.
    Opiniones = rng.random(N) # Asigno valores de opinión entre 0 y 1 de forma aleatoria
    return Opiniones

#---------------------------------------------------------------------------------------

def Evolucion_sistema(Opiniones,N,epsilon,mu):
    rng = np.random.default_rng() # Objeto de numpy que genera distribuciones números aleatorios y sampleos.
    
    # Elijo los dos agentes que van a interactuar
    Agentes = rng.choice(np.arange(N),size=2,replace=False)
    # Los agentes interactúan si sus opiniones se encuentran suficientemente cerca
    distancia = Opiniones[Agentes[1]]-Opiniones[Agentes[0]] # Distancia entre las opiniones de los agentes
    if np.abs(distancia) < epsilon:
        Interaccion_agentes(Opiniones,epsilon,mu)
        
        
# Esto lo pondré en una función aparte que voy a llamar interacción
# # Evoluciono al primer agente
# Opiniones[Agentes[0]] = Opiniones[Agentes[0]] + mu*(distancia)
# # Evoluciono al segundo agente
# Opiniones[Agentes[1]] = Opiniones[Agentes[1]] - mu*(distancia)
# El segundo tiene un menos porque quiero -diferencia, cosa de 
# que el segundo agente se acerque al primero
