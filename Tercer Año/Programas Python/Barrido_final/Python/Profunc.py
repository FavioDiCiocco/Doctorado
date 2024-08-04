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
"""

# Voy a ver de levantar datos de los archivos nuevos

# Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
# Distribución final
# Semilla
# Fragmentos Matriz de Adyacencia

# Levanto los datos del archivo
Datos = ldata("../Beta-Cosd/Opiniones_N=10000_kappa=10_beta=0.40_cosd=0.00_Iter=0.file")

puntos_medios = (np.linspace(-1,1,43)[0:-1] + np.linspace(-1,1,43)[1:])/2

dist_simul = np.reshape(np.array(Datos[1],dtype="float"),(42,42))

Opiniones = np.zeros(2*10000)
agregados = 0

for fila in range(dist_simul.shape[0]):
    for columna in range(dist_simul.shape[1]):
        
        cant_agregar = round(dist_simul[fila,columna] * 10000)
        if (cant_agregar > 0):
            x_i = puntos_medios[fila]
            y_i = puntos_medios[columna]
            
            vector_agregar = np.zeros(cant_agregar*2)
            vector_agregar[0::2] = np.ones(cant_agregar)*x_i
            vector_agregar[1::2] = np.ones(cant_agregar)*y_i
            
            Opiniones[agregados*2:(cant_agregar+agregados)*2] = vector_agregar
            
            agregados += cant_agregar
        
print(agregados)

"""

#####################################################################################

# Recorro las carpetas con datos
CarpCheck=[[root,files] for root,dirs,files in os.walk("../Beta-Cosd")]

# Me armo una lista con los nombres de todos los archivos con datos.
# Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos

Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]

for nombre in Archivos_Datos:
    
    file_path = Path("../Beta-Cosd/{}".format(nombre))
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Modify the second line
    lines[1] = lines[1].rstrip('\n') + '\t\n'
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

func.Tiempo(t0)
