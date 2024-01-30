#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:52:24 2023

@author: favio
"""
import numpy as np
import networkx as nx
import time

t0=time.time()

##################################################################################
##################################################################################

# FUNCIONES BASE

##################################################################################
##################################################################################

def scan(cant,lista):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
def Tiempo():
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))


def ldata(archive):
        f = open(archive)
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data 


# CREACIÓN REDES

##################################################################################
##################################################################################

# Voy a crear redes de Erdós-Renyi. Las redes van a ser de 1000 agentes,
# tengo que comprobar que las redes sean conexas y estudiar su matriz de Adyacencia.
# Así que la idea es armar un programa que construya redes con grado medio 8


rng = np.random.default_rng() # Objeto de numpy que genera distribuciones
N = 1000

for gmedio in range(4,11):
    for elemento in range(100):
        probabilidad = gmedio/(N-1)
        graph1 = nx.erdos_renyi_graph(n=N,p=probabilidad)
        # graph1 = nx.random_regular_graph(4,N)
    
        #--------------------------------------------------------------------------------------------------
    
        # Ahora me encargo de forzar la red a ser conexa. Para eso miro cuantas componentes conectadas hay.
        # Si hay más de una, primero me construyo una lista con todos los conjuntos, colocando el más grande de todos
        # al principio. Después, los enlaces que voy a agregar, tomando un sujeto al azar de la componente gigante
        # y un sujeto al azar de esos conjuntos libres que quedaron. Por otro lado, me armo una lista de enlaces
        # a remover, tomados de los enlaces de la componente Gigante original. Definidos los enlaces que voy a
        # agregar y los que voy a remover, los agrego y remuevo. Si me quedó todo conectado, genial, avanzo.
        # Sino, vuelvo a hacer el mismo proceso hasta que me quede todo conectado.
    
        while( not nx.is_connected(graph1) ):
            # Me armo la lista de componentes colocando primero al más grande
            Componentes = [list(c) for c in sorted(nx.connected_components(graph1), key=len, reverse=True)]
    
            # Armo las listas de tuplas de enlaces para agregar y remover
    
            Enlaces_Agregar = [(rng.choice(Componentes[0]),rng.choice(conjunto)) for conjunto in Componentes[1::]]
            Enlaces_Remover = rng.choice(list(graph1.edges(Componentes[0])),size = len(Componentes[1::]))
    
            # Agrego y remuevo enlaces
    
            graph1.add_edges_from(Enlaces_Agregar)
            graph1.remove_edges_from(Enlaces_Remover)
    
        #----------------------------------------------------------------------------------------------------
    
        # Una vez que tengo la red correctamente armada, ahora necesito obtener la matriz de adyacencia
        # como algo que después pueda pasar a C.
        
        filename = "../Programas C/MARE/Erdos-Renyi/gm={}/ErdosRenyi_N={}_ID={}.file".format(gmedio,N,elemento)
        
        with open(filename, "w") as file:
            file.write(f"{len(graph1.nodes)} {len(graph1.edges)}\n")
            for edge in graph1.edges():
                file.write(f"{edge[0]} {edge[1]}\n")
        
        # Adyacencia = nx.to_numpy_matrix(graph1)
        # np.savetxt("../Programas C/MARE/Erdos-Renyi/ErdosRenyi_N={}_ID={}.file".format(N,elemento),Adyacencia,fmt = "%d", delimiter = "\t", newline = "\t")

# Con esto me guardo la matriz como un txt con una única fila y todos los elementos son enteros.

    
Tiempo()