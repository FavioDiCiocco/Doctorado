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
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()


Datos = func.ldata("../Datos/Opiniones_N=1000_kappa=10.0_beta=0.50_cosd=0.00_Iter=0.file")

print(Datos[6])
print(type(Datos[7][0]))


func.Tiempo(t0)
