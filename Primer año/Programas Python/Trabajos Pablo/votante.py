#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:25:34 2022

@author: favio
"""

# Importo los paquetes necesarios para las funciones

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

##################################################################################
##################################################################################

# FUNCIONES

##################################################################################
##################################################################################

# Esta función recibe los parámetros del sistema, que son el largo de la grilla
# y la fracción de individuos con postura 1. A partir de eso arma la grilla y
# asigna las posturas a los agentes. Luego devuelve la grilla creada.

def Construccion_grilla_cuadrada(L,p):
    G = nx.grid_2d_graph(L,L) # Me armo una grilla de L*L agentes
    Valores = dict() # Este es el diccionario con los valores que le asigno a los nodos
    Dist0 = np.sign(-np.random.rand(L*L)+p) # Esta es la distribución inicial de posturas de los agentes.
    
    # Asocio cada nodo con su postura
    for nodo,postura in zip(G.nodes(),Dist0):
        Valores[nodo] = postura
        
    nx.set_node_attributes(G,Valores, name = "Postura") # Le asigno sus posturas a los agentes
    
    return G

#--------------------------------------------------------------------------------

# Esta función toma el estado actual del sistema, lo grafica y guarda el archivo
# en la carpeta indicada. Por ahora guardaré los archivos de imágenes y desde
# afuera iré armando gifs o revisando eso. Después veré de crear un plot animado
# o que se sobreescriba cuando todo lo demás funque.

def Graficar_y_guardar_sistema(G,L,iteracion,path="../../Imagenes/Trabajos Pablo"):
    fig,ax = plt.subplots(figsize=(12,8)) # Creo la figura que voy a graficar

    x = np.arange(L) # Largo de mi grilla
    y = np.arange(L) # Alto de mi grilla

    # Armo un grid de posturas de los agentes. 
    Z = np.zeros((L,L))
    for objeto in nx.get_node_attributes(G,"Postura").items():
        Z[objeto[0][0],objeto[0][1]] = objeto[1]

    # Grafico
    ax.pcolormesh(x,y,Z,shading="nearest",cmap="bwr")
    ax.set_title("Gráfico a tiempo t={}".format(iteracion))
    plt.savefig( path+"/Sistema_t={}.png".format(iteracion), bbox_inches = "tight")
    plt.close()
    
#--------------------------------------------------------------------------------

# Recibo la grilla del sistema y evoluciono su estado al elegir dos agentes al azar,
# el primero lo elijo del total de agentes, el segundo lo elijo del grupo de vecinos
# del primero. Luego copio la postura del primer agente al segundo agente.

def Imitacion_postura(G):
    rng = np.random.default_rng() # Objeto de numpy que genera distribuciones
    # números aleatorios y sampleos.
    
    Nodos = G.nodes() # Defino la lista de nodos de mi grilla
    nodo_i = tuple(rng.choice(Nodos)) # Elijo mi primer nodo al azar
    Vecinos = [nodo for nodo in nx.neighbors(G,nodo_i)] # Armo la lista de vecinos del nodo_i
    nodo_j = tuple(rng.choice(Vecinos)) # Elijo un vecino del nodo_i al azar
    
    # Armo un dict para que el cambio de postura se realice de forma más prolija.
    Cambio = dict()
    Cambio[nodo_j] = nx.get_node_attributes(G,"Postura")[nodo_i]
    
    nx.set_node_attributes(G,Cambio, name = "Postura") # Fijo la opinión del vecino j igual a la del vecino i.
    
#--------------------------------------------------------------------------------

def Enlaces_activos(G):
    Atributos = nx.get_node_attributes(G, "Postura") # Me guardo un diccionario de las posturas de cada agente
    
    # Lo siguiente es armar una lista con el producto de posturas entre todos los agentes que tengan
    # un enlace. Si sus posturas son contrarias, entonces el resultado es -1. Por tanto, cuento todos
    # los -1 obtenidos. activos es el número de enlaces cuyos extremos tienen posturas distintas
    activos = [Atributos[i]*Atributos[j] for i,j in G.edges()].count(-1)
    fraccion_activos = activos/len(G.edges()) # Esta es la fracción de enlaces activos del sistema
    return fraccion_activos
    