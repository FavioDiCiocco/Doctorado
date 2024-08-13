#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

"""

#####################################################################################
#####################################################################################

# Armo el gráfico de las regiones del espacio de parámetros Beta-Kappa

tlinea = 6

# Create a figure and axis
plt.rcParams.update({'font.size': 44})
fig, ax = plt.subplots(figsize=(28,21))

# Región de Consenso Neutral
x = [0, 1, 1, 0, 0]  # x-coordinates
y = [0, 0, 1.5, 1.5, 0]  # y-coordinates
# ax.fill(x, y, color='tab:gray')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label='Consenso Neutral')
ax.text(0.5, 1, 'I', fontsize=40, ha='center', va='center', color='k')

# Región de Consenso Radicalizado
x = np.concatenate((np.array([1,10]),np.flip(np.arange(10)+1)))  # x-coordinates
curva = np.exp((-1/19)*np.log(20)*np.arange(10))
y = np.concatenate((np.array([0,0]),np.flip(curva))) # y-coordinates
# ax.fill(x, y, color='tab:green')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label='Consenso Radicalizado')
ax.text(5, 0.25, 'II', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 1
x = np.concatenate((np.arange(3,11),np.array([10,3])))  # x-coordinates
y = np.concatenate((curva[2:],np.array([curva[2],curva[2]]))) # y-coordinates
# ax.fill(x, y, color='tab:blue')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (40~80%) y P1Da (20~40%)')
ax.text(8, 0.55, 'III', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 2
x = np.concatenate((np.arange(0,3)+1,np.array([10,10,1])))  # x-coordinates
y = np.concatenate((curva[0:3],np.array([curva[2],1.1,1.1]))) # y-coordinates
# ax.fill(x, y, color='tab:purple')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (60~80%), P1D (20~30%) y PDa (10%)')
ax.text(6, 0.9, 'IV', fontsize=40, ha='center', va='center', color='k')

# Región de Polarización Descorrelacionada
x = [1, 10, 10, 1]  # x-coordinates
y = [1.1, 1.1, 1.5, 1.5]  # y-coordinates
# ax.fill(x, y, color='tab:orange')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label='Polarización Descorrelacionada')
ax.text(5, 1.3, 'V', fontsize=40, ha='center', va='center', color='k')


ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$\beta$")
ax.set_title("Distribución de estados en el espacio de parámetros")
ax.set_xlim(0,10)
ax.set_ylim(0,1.5)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
# ax.legend()

direccion_guardado = Path("../../../Imagenes/Barrido_final/Beta-Kappa/Distribucion de estados.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


func.Tiempo(t0)
