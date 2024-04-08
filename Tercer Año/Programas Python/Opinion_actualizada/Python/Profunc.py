#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from pathlib import Path
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()


# Datos = func.ldata("../Datos/Opiniones_N=1000_kappa=10.0_beta=0.50_cosd=0.00_Iter=0.file")

# print(Datos[6])
# print(type(Datos[7][0]))

########################################################################################

"""
# Sample data
categories = ['Consenso Radicalizado', 'Polarización Ideológica con anchura', 'Polarización Ideológica sin anchura']
fracciones = np.array([[0.91,0.63,0.91,0.97],[0.09,0.37,0.09,0],[0,0,0,0.03]])
X = ["0.2", "0.4", "0.6", "0.8"]

direccion_guardado = Path("../../../Imagenes/Opinion_actualizada/Beta-Cosd/Fraccion estados barras.png")

plt.rcParams.update({'font.size': 44})
plt.figure("FracStacked",figsize=(28,21))


# Plot stacked bars
for fila in range(fracciones.shape[0]):
    if fila == 0:
        plt.bar(X, fracciones[fila],width = 0.3 ,label=categories[fila])
    else:
        plt.bar(X, fracciones[fila],bottom = np.sum(fracciones[0:fila], axis=0),width = 0.3 ,label=categories[fila])

# Add labels and title
plt.xlabel(r'Cos($\delta$)')
plt.ylabel('Fracción')
plt.title(r'Distribución de 100 estados, $\kappa$ = 10, Cos($\delta$) = 0.6')
plt.legend(loc='lower left', framealpha=0.5)
plt.grid()
plt.savefig(direccion_guardado , bbox_inches = "tight")
plt.close("FracStacked")

"""

#####################################################################################
#####################################################################################

# Armo el gráfico de las regiones del espacio de parámetros Beta-Kappa
"""
tlinea = 6

# Create a figure and axis
plt.rcParams.update({'font.size': 44})
fig, ax = plt.subplots(figsize=(28,21))

# Región de Consenso Neutral
x = [0, 1, 1, 0, 0]  # x-coordinates
y = [0, 0, 2, 2, 0]  # y-coordinates
# ax.fill(x, y, color='tab:gray')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label='Consenso Neutral')
ax.text(0.5, 1, 'I', fontsize=40, ha='center', va='center', color='k')

# Región de Polarización Descorrelacionada
x = [1, 20, 20, 1]  # x-coordinates
y = [1.1, 1.1, 2, 2]  # y-coordinates
# ax.fill(x, y, color='tab:orange')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label='Polarización Descorrelacionada')
ax.text(10, 1.5, 'II', fontsize=40, ha='center', va='center', color='k')

# Región de Consenso Radicalizado
x = np.concatenate((np.array([1,20]),np.flip(np.arange(20)+1)))  # x-coordinates
curva = np.exp((-1/19)*np.log(4)*np.arange(20))
y = np.concatenate((np.array([0,0]),np.flip(curva))) # y-coordinates
# ax.fill(x, y, color='tab:green')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label='Consenso Radicalizado')
ax.text(10, 0.25, 'III', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 1
x = np.concatenate((np.arange(8,21),np.array([20,8])))  # x-coordinates
y = np.concatenate((curva[7:],np.array([0.6,0.6]))) # y-coordinates
# ax.fill(x, y, color='tab:blue')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (40~80%) y P1Da (20~40%)')
ax.text(17, 0.45, 'IV', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 2
x = np.concatenate((np.arange(0,8)+1,np.array([20,20,1])))  # x-coordinates
y = np.concatenate((curva[0:8],np.array([curva[7],1.1,1.1]))) # y-coordinates
# ax.fill(x, y, color='tab:purple')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (60~80%), P1D (20~30%) y PDa (10%)')
ax.text(10, 0.85, 'V', fontsize=40, ha='center', va='center', color='k')

ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$\beta$")
ax.set_title("Distribución de estados en el espacio de parámetros")
ax.set_xlim(0,20)
ax.set_ylim(0,2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
# ax.legend()

direccion_guardado = Path("../../../Imagenes/Opinion_actualizada/Beta-Kappa/Distribucion de estados.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

"""
#####################################################################################
#####################################################################################

# Armo el gráfico de las regiones del espacio de parámetros Beta-Cosd

tlinea = 6

# Create a figure and axis
plt.rcParams.update({'font.size': 44})
fig, ax = plt.subplots(figsize=(28,21))

# Región de Consenso Radicalizado
x = [0, 1, 1, 0.8, 0.6, 0.2, 0.3, 1, 1, 0.1, 0.1, 0, 0]  # x-coordinates
y = [0, 0, 0.15, 0.2, 0.25, 0.3, 0.6, 0.6, 1.1, 1.1, 0.3, 0.2, 0]  # y-coordinates
# ax.fill(x, y, color='tab:green')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)

# Región de Polarización Descorrelacionada
x = [0, 0.1, 0.2, 0.2, 0, 0]  # x-coordinates
y = [1.1, 1.1, 1.3, 2, 2, 1.1]  # y-coordinates
# ax.fill(x, y, color='tab:orange')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) #, label='Polarización Descorrelacionada')

# Región de Transición
x = [0.1, 0.2, 0.3, 0.4, 0.4, 0.2]  # x-coordinates
y = [1.1, 1.1, 1.3, 1.5, 2, 2]  # y-coordinates
# ax.fill(x, y, color='tab:olive')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)

# Región de Polarización ideológica
x = [0.2, 1, 1, 0.4, 0.4, 0.3] # x-coordinates
y = [1.1, 1.1, 2, 2, 1.5, 1.3] # y-coordinates
# ax.fill(x, y, color='tab:red')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)

# Región de Mezcla 1
x = [0.2, 0.6, 0.8, 1, 1, 0.3] # x-coordinates
y = [0.3, 0.25, 0.2, 0.15, 0.6, 0.6] # y-coordinates
# ax.fill(x, y, color='tab:blue')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (40~80%) y PIa (10~45%)')

# Región de Mezcla 2
x = [0, 0.1, 0.1, 0] # x-coordinates
y = [0.2, 0.3, 0.7, 0.7] # y-coordinates
# ax.fill(x, y, color='tab:purple')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)  # label=r'Mezcla: CR (50~80%), P1Da (20~35%)')

# Región de Mezcla 3
x = [0, 0.1, 0.1, 0, 0] # x-coordinates
y = [0.7, 0.7, 1.1, 1.1, 0.2] # y-coordinates
# ax.fill(x, y, color='tab:brown')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (30~40%), P1D (10~50%)')

ax.set_xlabel(r"$cos(\delta)$")
ax.set_ylabel(r"$\beta$")
ax.set_title("Distribución de estados en el espacio de parámetros")
ax.set_xlim(0,1)
ax.set_ylim(0,2)
# ax.legend()

direccion_guardado = Path("../../../Imagenes/Opinion_actualizada/Beta-Cosd/Distribucion de estados.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


func.Tiempo(t0)
