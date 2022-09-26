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

##################################################################################
##################################################################################

# FUNCIONES GENERALES

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------

# Esto printea una cantidad de valores cant de un objeto iterable que paso
# en la parte de lista.
def scan(lista,cant=10):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
#--------------------------------------------------------------------------------
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo(t0):
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))

#--------------------------------------------------------------------------------

##################################################################################
##################################################################################

# CÓDIGO PRINCIPAL

##################################################################################
##################################################################################

# Definición de los parámetros del modelo
#---------------------------------------------------------------

L = 4 # Ancho y largo de la grilla
p = 0.5 # Distribución de los agentes positivos y los negativos

# Armado de la red
#---------------------------------------------------------------

# Voy a probar esta función de networkx que genera una grilla 2D.
# Podría hacer esto por mi cuenta usando un array de Numpy, pero me inclino a
# pensar que si Networkx decidió armar grillas por sí mismo en vez de usar
# arrays, alguna ventaja debe tener esto

G = nx.grid_2d_graph(L,L) # Me armo una grilla de L*L agentes

# Ahora tengo que agregar a cada uno de los nodos la característica de su estado.

Valores = dict() # Este es el diccionario con los valores que le asigno a los nodos

# Esta es la distribución inicial de posturas de los agentes. Lo importante
# es que de esta forma puedo modificar para que la distribución inicial tenga
# la fracción p de agentes con postura 1.
Dist0 = np.sign(-np.random.rand(L*L)+p)

# Asocio cada nodo con su postura
for nodo,postura in zip(G.nodes(),Dist0):
    Valores[nodo] = postura

nx.set_node_attributes(G,Valores, name = "Postura") # Le asigno sus posturas a los agentes

print(nx.get_node_attributes(G, "Postura"))

# Evolución del sistema
#---------------------------------------------------------------

# La idea es que tomo un agente al azar, luego tomo un vecino suyo 
# al azar y de ahí pido que el primer vecino tenga la opinión del segundo.

Nodos = np.array(G.nodes())

# Al parecer hubo un cambio en las librerías de numpy y lo que se recomienda
# es no usar randint, sino usar los generadores de números aleatorios en
# el código nuevo. Para esto entonces la idea es primero crear un generador
# de números aleatorios.

rng = np.random.default_rng() # Esto es un objeto, es un generador de números aleatorios.
# Este sujeto me permite generar ints, distribuciones normales y otras cosas también.

# Tomo mi primer nodo al azar.

nodo_i = tuple(rng.choice(Nodos)) # Choice toma un elemento al azar del array

# Ahora elijo un vecino de ese nodo.

Vecinos = [nodo for nodo in nx.neighbors(G,nodo_i)]
nodo_j = tuple(rng.choice(Vecinos))

# Fijo la opinión del vecino j igual a la del vecino i.

nx.set_node_attributes(G, {nodo_j : nx.get_node_attributes(G,"Postura")[nodo_i]}, name = "Postura")

print(nodo_i,nodo_j)
print(nx.get_node_attributes(G, "Postura"))