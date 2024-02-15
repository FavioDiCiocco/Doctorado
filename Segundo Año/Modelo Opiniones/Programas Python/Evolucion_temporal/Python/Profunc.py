#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pathlib import Path
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

# Grados = np.zeros(3)

# Cargo el archivo con la matriz de adyacencia Erdos-Renyi
for elemento in np.array([40,51,66]):
    Datos = func.ldata("../MARE_Algarve/Erdos-Renyi/ErdosRenyi_N=1000_ID={}.file".format(elemento))
    Adyacencia  = np.reshape(np.array([i for i in Datos[0][:-1:]],dtype = "int"),(N,N))
    
    # Armo el grafo a partir de la matriz de Adyacencia
    
    G = nx.from_numpy_matrix(Adyacencia)
    
    Vecinos = [nodo for nodo in G[79]]
    
    
    gradomedio = 0
    for nodo in G.nodes():
        gradomedio += G.degree[nodo]
    
    # Grados[elemento] = gradomedio / N
    print(gradomedio / N)

# plt.hist(Grados)

###################################################################################################
"""
# Dada una red, me armo un archivo como los que tiene Hugo de enlaces del sistema.

filename = "ER1000k=8.file"

with open(filename, "w") as file:
    file.write(f"{len(G.nodes)} {len(G.edges)}\n")
    for edge in G.edges():
        file.write(f"{edge[0]} {edge[1]}\n")


###################################################################################################

# Acá voy a mirar el tema del peso y la distancia entre los agentes 79 y 388

Datos = func.ldata("../2D_dtchico/Testigos_N=1000_kappa=10.0_beta=0.90_cosd=0.00_Iter=51.file")
Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
for i,fila in enumerate(Datos[1:-1:]):
    Testigos[i] = fila[:-1]

# Con esto tengo los datos de todos los agentes a lo largo del tiempo

Distancia = np.zeros(Testigos.shape[0])
Peso = np.zeros(Testigos.shape[0])

for nt in range(Testigos.shape[0]):
    
    # Calculo la distancia entre los agentes 79 y 388
    diferencia = Testigos[nt,[79*2,79*2+1]]-Testigos[nt,[388*2,388*2+1]]
    Distancia[nt] = np.linalg.norm(diferencia)

    # Calculo el peso entre los agentes 79 y 388
    numerador = pow(np.linalg.norm(diferencia)+0.002*10,-0.9)
    
    denominador = 0
    for nodo in Vecinos:
        diferencia = Testigos[nt,[79*2,79*2+1]]-Testigos[nt,[nodo*2,nodo*2+1]]
        denominador += pow(np.linalg.norm(diferencia)+0.002*10,-0.9)
    
    Peso[nt] = numerador/denominador

# Normalizo la distancia
# Distancia = Distancia / np.max(Distancia)

###################################################################################################

# Armo el gráfico con la distancia y el peso

direccion_guardado = Path("../../../Imagenes/Evolucion_temporal/2D_dtchico/Peso (79 388) vs T.png")

plt.rcParams.update({'font.size': 32})
plt.figure("PesoDist",figsize=(20,15))
X = np.arange(Testigos.shape[0])*0.001
plt.plot(X,Peso, color = "tab:blue", label = "Peso" ,linewidth = 6)
# plt.plot(X,Distancia, color = "tab:green", label = "Distancia" ,linewidth = 6)
# plt.axvline(10.25, color = "tab:red" ,linestyle = "--", linewidth = 3)
plt.xlabel(r"Tiempo$(10^{-3})$")
plt.grid(alpha = 0.8)
plt.legend()
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close("PesoDist")


###################################################################################################

Vecinos_cercanos = [112,374,633,968]
Pesos_cercanos = np.zeros((Testigos.shape[0],len(Vecinos_cercanos)))

Vecinos_lejanos = [5,122,388]
Pesos_lejanos = np.zeros((Testigos.shape[0],len(Vecinos_lejanos)))
Distancia = np.zeros(Testigos.shape[0])



for nt in range(Testigos.shape[0]):
    
    denominador = 0
    for nodo in Vecinos:
        diferencia = Testigos[nt,[79*2,79*2+1]]-Testigos[nt,[nodo*2,nodo*2+1]]
        denominador += pow(np.linalg.norm(diferencia)+0.002*10,-0.9)
    
    for i,nodo in enumerate(Vecinos_cercanos):
        diferencia = Testigos[nt,[79*2,79*2+1]]-Testigos[nt,[nodo*2,nodo*2+1]]
        numerador = pow(np.linalg.norm(diferencia)+0.002*10,-0.9)
        Pesos_cercanos[nt,i] = numerador/denominador
        
    for i,nodo in enumerate(Vecinos_lejanos):
        diferencia = Testigos[nt,[79*2,79*2+1]]-Testigos[nt,[nodo*2,nodo*2+1]]
        numerador = pow(np.linalg.norm(diferencia)+0.002*10,-0.9)
        Pesos_lejanos[nt,i] = numerador/denominador

direccion_guardado = Path("../../../Imagenes/Evolucion_temporal/2D_dtchico/Pesos opuestos vs T.png")

plt.rcParams.update({'font.size': 32})
plt.figure("Pesos_opuestos",figsize=(20,15))
X = np.arange(Testigos.shape[0])*0.001
plt.plot(X,np.sum(Pesos_cercanos,axis=1), color = "tab:blue", label = "Agentes Cercanos" ,linewidth = 3)
plt.plot(X,np.sum(Pesos_lejanos,axis=1), color = "tab:green", label = "Agentes Lejanos" ,linewidth = 3)
# plt.plot(X,Distancia, color = "tab:green", label = "Distancia" ,linewidth = 6)
#plt.axvline(10.75, color = "tab:red" ,linestyle = "--", linewidth = 3)
plt.xlabel(r"Tiempo$(10^3)$")
plt.grid(alpha = 0.8)
plt.legend()
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close("Pesos_opuestos")

###################################################################################################


# Acá voy a comparar la simulación que armé yo con la simulación de Hugo, para
# ver qué puede ser lo que esté trabajando distinto y resultando en que los códigos
# no dan exactamente iguales al final. Quizás es una cosa de que no están corriendo
# la misma cantidad exacta de pasos y mi código está en un punto distinto de oscilación.

Datos_mios = func.ldata("../1D/Testigos_N=1000_kappa=10.0_beta=0.90_cosd=0.00_Iter=51.file")
Datos_Hugo = func.ldata("../1D_Hugo/Testigos_N=1000_kappa=10_beta=0.9_cosd=0_Iter=51.file")

T = np.arange(2000)*0.01
Diferencia = np.zeros(2000)

for fila in range(1,2001):
    Opiniones_mios = np.array([float(x) for x in Datos_mios[fila][:-1]])
    Opiniones_Hugo = np.array([float(x) for x in Datos_Hugo[fila][:-1]])
    
    Diferencia[fila-1] = np.linalg.norm(Opiniones_mios-Opiniones_Hugo)

direccion_guardado = Path("../../../Imagenes/Evolucion_temporal/1D/Distancia_simulaciones.png")

plt.rcParams.update({'font.size': 32})
plt.figure("Distancia_sistemas",figsize=(20,15))
plt.plot(T,Diferencia,linewidth = 6)
plt.xlabel(r"Tiempo$(10^3)$")
plt.title("Distancia entre estados")
plt.grid(alpha = 0.8)
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close("Distancia_sistemas")

"""

func.Tiempo(t0)

