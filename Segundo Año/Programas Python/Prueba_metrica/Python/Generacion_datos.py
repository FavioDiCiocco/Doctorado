# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:32:31 2023

@author: favio
"""

# Voy a armar los datos que me generan mis diversas configuraciones. Esas
# configuraciones luego las voy a usar para probar las métricas de medir opiniones.

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import time
import math

##################################################################################
##################################################################################

# FUNCIONES GENERALES

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo(t0):
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))

#--------------------------------------------------------------------------------

def Guardar_archivo(array, filename):
    # Write text to the file
    with open(filename, 'w') as file:
        file.write("Opinion Inicial del sistema")
        file.write('\n')  # Add some spacing between text and array
        file.write("Relleno")
        file.write('\n')  # Add some spacing between text and array
        file.write("Variación promedio")
        file.write('\n')  # Add some spacing between text and array
        file.write("Relleno")
        file.write('\n')  # Add some spacing between text and array
        file.write("Opiniones Finales")
        file.write('\n')  # Add some spacing between text and array
    
    # Append the array to the file
    with open(filename, 'a') as file:
        array_str = '\t'.join(map(str, array))
        file.write(array_str)
    
    print(f"The text and array have been saved to {filename}")


##################################################################################
##################################################################################
    
rng = np.random.default_rng()

##################################################################################
# Armo primero el gráfico del consenso de opiniones. Los valores de X e Y
# son la distribución de opiniones en tópico 1 y 2

media = 0
desv = 0.05
N = 1000

X = rng.normal(media,desv,N)
Y = rng.normal(media,desv,N)

Opiniones = np.zeros(2*N)
Opiniones[0::2] = X
Opiniones[1::2] = Y

filename = "../Datos/Opiniones_N=1000_kappa=0.2_beta=1_cosd=0.00_Iter=0.file"

Guardar_archivo(Opiniones,filename)