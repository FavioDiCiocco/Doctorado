#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import numpy as np
from pathlib import Path
import os
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

#####################################################################################

# Voy a tomar los archivos que tengo de la simulación en la región de Beta [0,1]
# y Cosd [0,0.15] y modificar los datos de salida para que sean iguales a los datos
# que construí en las regiones extendidas. Esos datos tienen la distribución de las
# opiniones respecto de una grilla de 42*42.

def Clasificacion(Array, Nx, Ny,T):
    
    # Recibo un array de opiniones que van entre [-1,1]. Le sumo 1
    # para que las opiniones vayan entre [0,2].
    Array = Array+1
    
    # Divido mi espacio de tópicos 2D en cuadrados. Defino el ancho
    # de esos cuadrados.
    ancho_x = 2/Nx
    ancho_y = 2/Ny
    
    # Armo un array de tuplas que indiquen "fila" y "columna" en la cuál
    # cae cada opinión.
    Ubicaciones = np.array([(math.floor(x/ancho_x),math.floor(y/ancho_y)) for x,y in zip(Array[0::T],Array[1::T])])
    
    # Ahora me armo mi array de distribución, que cuenta cuántas opiniones tengo
    # por cada cajita.
    Distribucion = np.zeros((Nx*Ny))
    for opinion in Ubicaciones:
        # Tomo mínimos para que no intente ir a una cajita no existente. Tendría un problema
        # si algún agente tiene opinión máxima en algún tópico.
        fila = min(opinion[0],Nx-1)
        columna = min(opinion[1],Ny-1)
        Distribucion[fila*Ny+columna] += 1
    
    # Una vez armada mi distribucion, la normalizo.
    Distribucion = Distribucion/np.sum(Distribucion)
    
    # Returneo la distribucion
    return Distribucion

#####################################################################################

# Esta es la función que uso por excelencia para levantar datos de archivos. Lo
# bueno es que lee archivos de forma general, no necesita que sean csv o cosas así
def ldata(archive):
    with open(archive) as f:
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data

#####################################################################################

# La función de Clasificación la necesito para armar mi distribución de opiniones.

# Recorro las carpetas con datos
Direccion = Path("../Descarga")
CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]

for nombre in CarpCheck[0][1]:
    
    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
    # Opinión Inicial del sistema
    # Variación Promedio
    # Opinión Final
    # Pasos simulados
    # Semilla
    # Matriz de Adyacencia
    
    # Levanto los datos del archivo
    Datos = ldata( Direccion / nombre)
    
    # Leo los datos de las Opiniones Finales y me armo una distribución en forma de matriz de 7x7
    Opifinales = np.array(Datos[5][:-1], dtype="float")
    Opifinales = Opifinales / 10
    Distr_Sim = Clasificacion(Opifinales,42,42,2)
    
    # Una vez que tengo armada la distribución a partir de las opiniones finales,
    # paso mis datos a los nuevos archivos imitando el nuevo formato.
    
    # Distribución final
    # Semilla
    # Matriz de Adyacencia
    
    with open('../Beta-Cosd/{}'.format(nombre), 'w') as file:
        
        # Write the string
        file.write("Distribución final\n")
        np.savetxt(file, np.reshape(Distr_Sim,(1,Distr_Sim.shape[0])),fmt ="%.6f",delimiter="\t")
        
        file.write("Semilla\n")
        file.write(Datos[9][0] + "\n")
        
        file.write("Primeras filas de la Matriz de Adyacencia\n")
        for i in range(11,len(Datos)):
            A_vecinos = np.array(Datos[i][:-1], dtype="int")
            
            np.savetxt(file,np.reshape(A_vecinos,(1,A_vecinos.shape[0])),fmt = "%d",delimiter="\t")

func.Tiempo(t0)
