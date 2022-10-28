#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:08:53 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import time
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

N = 1000 # Número de agentes
mu = 0.5 # Parámetro de convergencia
epsilon = 0.1 # Tolerancia de opiniones

Opiniones = deff.Distribucion_inicial(N) # Armo la distribución inicial de opiniones
t = 0 # Este es el "paso temporal discreto" del sistema.

# Grafico la primer columna de datos
plt.rcParams.update({'font.size': 24}) # Defino los tamaños de letras
fig,ax = plt.subplots(figsize=(12,8)) # Creo la figura que voy a graficar
T = np.ones(N)*t # Largo de mi grilla
ax.plot(T,Opiniones,"xg",markersize=6) # Grafico el estado actual del sistema

criterio_corte = mu*epsilon*(3/2)*(1/5000) # Este es el criterio que definí
# para decir que el sistema está en un estado estable
desviacion_estandar = criterio_corte+1 # Fuerzo a que el while se ejecute una vez mínimo

# while desviacion_estandar > criterio_corte
for tiempos in range(100):
    Opiniones_previas = np.array([opi for opi in Opiniones]) # Armo el array de opiniones del paso actual
    for iteracion in range(N):
        deff.Evolucion_sistema(Opiniones, N, epsilon, mu) # Evoluciono las opiniones de los agentes de forma aleatoria
    t += 1 # Actualizo mi índice del tiempo
    # Grafico la siguiente columna de datos
    T = np.ones(N)*t # Largo de mi grilla
    ax.plot(T,Opiniones,"xg",markersize=4) # Grafico el estado actual del sistema
    # desviacion_estandar = np.std(np.abs(Opiniones-Opiniones_previas)) # Esta es la desviación estándar
    # de la diferencia entre el paso actual y el paso previo de las opiniones de los agentes

# Cuando el sistema alcanzó un estado estable, guardo mi gráfico
plt.savefig( "../../Imagenes/Trabajos Pablo/Deffuant_mu={}_eps={}.png".format(mu,epsilon), bbox_inches = "tight")
plt.close()
