#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import math
import time
import os
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

#--------------------------------------------------------------------------------

# Esta es la función que uso por excelencia para levantar datos de archivos. Lo
# bueno es que lee archivos de forma general, no necesita que sean csv o cosas así
def ldata(archive):
        f = open(archive)
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data 

#--------------------------------------------------------------------------------

def Clasificacion(Array,N,T):
    
    # Recibo un array de opiniones que van entre [-1,1]. Le sumo 1
    # para que las opiniones vayan entre [0,2].
    Array = Array+1
    
    # Divido mi espacio de tópicos 2D en cuadrados. Defino el ancho
    # de esos cuadrados.
    ancho = 2/N
    
    # Armo un array de tuplas que indiquen "fila" y "columna" en la cuál
    # cae cada opinión.
    Ubicaciones = np.array([(math.floor(x/ancho),math.floor(y/ancho)) for x,y in zip(Array[0::T],Array[1::T])])
    
    # Ahora me armo mi array de distribución, que cuenta cuántas opiniones tengo
    # por cada cajita.
    Distribucion = np.zeros((N*N))
    for opinion in Ubicaciones:
        # Tomo mínimos para que no intente ir a una cajita no existente. Tendría un problema
        # si algún agente tiene opinión máxima en algún tópico.
        fila = min(opinion[1],N-1)
        columna = min(opinion[0],N-1)
        Distribucion[fila*N+columna] += 1
    
    # Una vez armada mi distribucion, la normalizo.
    Distribucion = Distribucion/np.sum(Distribucion)
    
    # Returneo la distribucion
    return Distribucion

#-----------------------------------------------------------------------------------------------
    
# Transformo estas cosas en paths. Espero que acostumbrarme a esto valga la pena
Direccion = Path("../{}".format("Datos"))
carpeta = Path("Datos")

# Recorro las carpetas con datos
CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]

# Me armo una lista con los nombres de todos los archivos con datos.
# Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos

Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]

Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})
    
# Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[1].split("=")[1]))
Df_archivos["Kappas"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
Df_archivos["parametro_y"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
Df_archivos["parametro_x"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1]))
Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[5].split("=")[1].strip(".file")))



# Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
# me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
# me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

# Defino la cantidad de agentes de la red
AGENTES = int(np.unique(Df_archivos["n"]))

# Defino los arrays de parámetros diferentes
Arr_KAPPAS = np.unique(Df_archivos["Kappas"])[0:1]
Arr_param_x = np.unique(Df_archivos["parametro_x"])[0:1]
Arr_param_y = np.unique(Df_archivos["parametro_y"])[0:1]


# Armo una lista de tuplas que tengan organizados los parámetros a utilizar
Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
               for j,param_y in enumerate(Arr_param_y)]

# Defino el tipo de archivo del cuál tomaré los datos
TIPO = "Opiniones"

# Sólo tiene sentido graficar en dos dimensiones, en una es el 
# Gráfico de Opi vs T y en tres no se vería mejor.
T=2

path = Direccion
N = 20

Salida = dict()
for KAPPAS in Arr_KAPPAS:
    Salida[KAPPAS] = dict()
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(Df_archivos.loc[(Df_archivos["tipo"]==TIPO) & 
                                    (Df_archivos["n"]==AGENTES) & 
                                    (Df_archivos["Kappas"]==KAPPAS) & 
                                    (Df_archivos["parametro_x"]==PARAM_X) &
                                    (Df_archivos["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        entropias = np.zeros(archivos.shape[0])
        
        for nombre in archivos:
    
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
    
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
    
            # Leo los datos de las Opiniones Finales
            Opifinales = np.array(Datos[5], dtype="float")
            Opifinales = Opifinales / np.max(np.abs(Opifinales))
    
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
    
            repeticion = int(Df_archivos.loc[Df_archivos["nombre"]==nombre,"iteracion"])
            
            # Armo mi array de Distribucion, que tiene la proba de que una opinión
            # pertenezca a una región del espacio de tópicos
            Probas = Clasificacion(Opifinales,N,T)
            
            # Con esa distribución puedo directamente calcular la entropía.
            entropias[repeticion] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
    
        if PARAM_X not in Salida[KAPPAS].keys():
            Salida[KAPPAS][PARAM_X] = dict()
        Salida[KAPPAS][PARAM_X][PARAM_Y] = entropias/np.log2(N)

func.Tiempo(t0)
