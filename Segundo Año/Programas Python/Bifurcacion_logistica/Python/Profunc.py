#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import math
import time
from pathlib import Path
from cycler import cycler
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

# Voy a armar el gráfico en 3D de la región en la cual el sistema tiene tres
# puntos fijos. Voy a barrer en Epsilon y Alfa y obtener los Kappa asociados.

#------------------------------------------------------------------------------

# Defino las funciones que uso para calcular los puntos críticos y los Kappa

def Derivada_kappa(x,alfa,epsilon):
    return np.exp(alfa*x-epsilon)+1-alfa*x

def Kappa(x,alfa,epsilon):
    return x*( 1 + np.exp(-alfa*x +epsilon) )

#------------------------------------------------------------------------------
"""

# Preparo mis variables para graficar

Epsilons = np.linspace(2,5,50)
Alfas = np.linspace(0.5,5,50)
XX,YY = np.meshgrid(Epsilons,Alfas)

# Armos las dos matrices que formarán las superficies de Kappa que voy a graficar.

Kappas_min = np.zeros(XX.shape)
Kappas_max = np.zeros(XX.shape)

# Calculo los Kappa y armo las matrices

for fila,alfa in enumerate(Alfas):
    for columna,epsilon in enumerate(Epsilons):
        
        # Calculo dónde se encuentra el mínimo de mi función Derivada_Kappa
        x_min = epsilon/alfa
        
        # Calculo los puntos críticos donde voy a encontrar los Kappa máximos y mínimos
        raiz_min = fsolve(Derivada_kappa,x_min-3,args=(alfa,epsilon))
        raiz_max = fsolve(Derivada_kappa,x_min+3,args=(alfa,epsilon))
        
        # Asigno los valores de los Kappa a mis matrices
        Kappas_min[fila,columna] = Kappa(raiz_max, alfa, epsilon)
        Kappas_max[fila,columna] = Kappa(raiz_min, alfa, epsilon) 
        
#------------------------------------------------------------------------------------------

# Ya tengo mis tres matrices, ahora puedo armar el gráfico de mis superficies.

# Hago unos primeros ajustes generales al gráfico

plt.rcParams.update({'font.size': 32})
fig = plt.figure("Region_triple_puntofijo",figsize=(40,40))
ax = fig.add_subplot(projection = "3d")
ax.set_xlabel(r"$\epsilon$",labelpad = 30)
ax.set_ylabel(r"$\alpha$",labelpad = 30)
ax.set_zlabel(r"$\kappa$", labelpad = 50)


# Grafico las dos superficies que encierran la región de tres puntos fijos

surf1 = ax.plot_surface(XX,YY,Kappas_min, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

surf2 = ax.plot_surface(XX,YY,Kappas_max, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)


direccion_guardado = Path("../../../Imagenes/Bifurcacion_logistica/Region_triple_puntofijo_BASE.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")


ax.view_init(0,0,0)
direccion_guardado = Path("../../../Imagenes/Bifurcacion_logistica/Region_triple_puntofijo_FRENTE.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")

ax.view_init(0,-90,0)
direccion_guardado = Path("../../../Imagenes/Bifurcacion_logistica/Region_triple_puntofijo_LATERAL.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")

ax.view_init(90,0,0)
direccion_guardado = Path("../../../Imagenes/Bifurcacion_logistica/Region_triple_puntofijo_TECHO.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")

ax.view_init(-90,0,0)
direccion_guardado = Path("../../../Imagenes/Bifurcacion_logistica/Region_triple_puntofijo_PISO.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")



plt.close("Region_triple_puntofijo")


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Armo una función que grafica Kappa en función de Epsilon y me arma curvas para
# distintos alfas.

# Armo mis variables a graficar

Epsilons = np.linspace(2,5,50)
Alfas = np.array([2,3,4])

#-------------------------------------------------------------------------------

# Barro en Alfas, que serán la cantidad de gráficos que arme

for Alfa in Alfas:
    
    # Abro el gráfico y defino los nombres de los ejes
    plt.rcParams.update({'font.size': 24})
    plt.figure("Cortes_3D",figsize=(20,15))
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\kappa$")
    
    # Armo mi array donde pondré los valores de Kappa, tanto los máx como los mín
    Kappas_min = np.zeros(Epsilons.shape[0])
    Kappas_max = np.zeros(Epsilons.shape[0])
    
    for indice,epsilon in enumerate(Epsilons):
        
        # Calculo dónde se encuentra el mínimo de mi función Derivada_Kappa
        x_min = epsilon/Alfa
        
        # Calculo los puntos críticos donde voy a encontrar los Kappa máximos y mínimos
        raiz_min= fsolve(Derivada_kappa,x_min-3,args=(Alfa,epsilon))[0]
        raiz_max = fsolve(Derivada_kappa,x_min+3,args=(Alfa,epsilon))[0]
        
        # Asigno los valores de los Kappa a mis matrices
        Kappas_min[indice] = Kappa(raiz_max, Alfa, epsilon)
        Kappas_max[indice] = Kappa(raiz_min, Alfa, epsilon)
        
    # Ahora que tengo la curva hecha, la grafico y guardo el gráfico
    
    
    
    plt.plot(Epsilons,Kappas_min,"--g",label=r"$\alpha$ = {}".format(Alfa), linewidth = 8)
    plt.plot(Epsilons,Kappas_max,"--g", linewidth = 8)


    # Preparo los detalles finales del gráfico
    
    plt.grid(alpha = 0.5)
    plt.legend()
    direccion_guardado = Path("../../../Imagenes/Bifurcacion_logistica/Cortes en Alfa={}.png".format(Alfa))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close("Cortes_3D")
    


#------------------------------------------------------------------------------------------------

"""

func.Tiempo(t0)
