#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:40:50 2022

@author: favio
"""

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

L = 30 # Ancho y largo de la grilla
p = 0.5 # Distribución de los agentes positivos y los negativos

# Evolución del modelo
# ----------------------------------------------------------------

G = vote.Construccion_grilla_cuadrada(L,p) # Construyo la red y asigno las posturas
vote.Graficar_y_guardar_sistema(G,L,0) # Me guardo un gráfico del sistema.
# Se le puede asignar un path para que guarde el archivo en la carpeta que quiera
fraccion_activos = vote.Enlaces_activos(G) # Calculo la fracción actual de agentes activos.
# N = len(G.nodes())
t = 1 # Este es el "paso temporal discreto" del sistema.
while fraccion_activos > 0:
    for iteracion in range(L*L*L):
        vote.Imitacion_postura(G)
    vote.Graficar_y_guardar_sistema(G,L,t) # Me guardo el estaod del sistema a tiempo t
    fraccion_activos = vote.Enlaces_activos(G) # Calculo la fracción agentes activos a tiempo t.
    t += 1 # Avanzo el índice t
    
func.Tiempo(t0)
