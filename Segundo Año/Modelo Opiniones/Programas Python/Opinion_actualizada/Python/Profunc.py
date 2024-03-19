#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
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


func.Tiempo(t0)
