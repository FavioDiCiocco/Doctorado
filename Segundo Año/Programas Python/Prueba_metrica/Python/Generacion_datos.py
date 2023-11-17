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
# Los valores de X e Y son la distribución de opiniones en tópico 1 y 2


# Armo primero el gráfico del consenso de opiniones.

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

##################################################################################
# Armo los gráfico de polarización a uno de los extremos

desv = 0.05
N = 1000

Medias = np.array([-5,5])

for indice1,mediax in enumerate(Medias):
    for indice2,mediay in enumerate(Medias):

        X = rng.normal(mediax,desv,N)
        Y = rng.normal(mediay,desv,N)
        
        Opiniones = np.zeros(2*N)
        Opiniones[0::2] = X
        Opiniones[1::2] = Y
        
        filename = "../Datos/Opiniones_N=1000_kappa=6_beta=0.6_cosd=0.00_Iter={}.file".format(indice2+2*indice1)
        
        Guardar_archivo(Opiniones,filename)
        


##################################################################################

# Armo los gráfico de polarización a uno de los extremos con anchura.
# Quiero armar cuatro tipos de gráficos de estos. Bimodal, homogéneo,
# más grande de un extremo que del otro.

desv = 2
N = 1000

# Primero hago el caso homogéneo

Medias = np.array([-5,5])

for indice1,kx in enumerate(Medias):
    for indice2,ky in enumerate(Medias):

        X = rng.random(N)*kx
        Y = rng.random(N)*ky
        
        Opiniones = np.zeros(2*N)
        Opiniones[0::2] = X
        Opiniones[1::2] = Y
        
        filename = "../Datos/Opiniones_N=1000_kappa=6_beta=0.6_cosd=0.00_Iter={}.file".format(indice2+2*indice1+4)
        
        Guardar_archivo(Opiniones,filename)
        


# Segundo hago el caso bimodal

extremos = np.array([-1,1])
K = 5
desv = 0.5

for indice1,direcx in enumerate(extremos):
    for indice2,direcy in enumerate(extremos):

        X1 = rng.normal(1.5,desv,int(N/2))
        X2 = rng.normal(K-1.5,desv,int(N/2))
        
        X = np.concatenate((X1,X2))
        X[X < 0] = np.zeros(X[X < 0].shape[0])
        X[X > K] = np.ones(X[X > K].shape[0])*K
        
        Y1 = rng.normal(1.5,desv,int(N/2))
        Y2 = rng.normal(K-1.5,desv,int(N/2))
        
        Y = np.concatenate((Y1,Y2))
        Y[Y < 0] = np.zeros(Y[Y < 0].shape[0])
        Y[Y > K] = np.ones(Y[Y > K].shape[0])*K
        
        X = X*direcx
        Y = Y*direcy
        
        Opiniones = np.zeros(2*N)
        Opiniones[0::2] = X
        Opiniones[1::2] = Y
        
        filename = "../Datos/Opiniones_N=1000_kappa=6_beta=0.6_cosd=0.00_Iter={}.file".format(indice2+2*indice1+8)
        
        Guardar_archivo(Opiniones,filename)
        

# Tercero hago el caso más grande de un extremo

extremos = np.array([-1,1])
K = 5
desv = 0.25

for indice1,direcx in enumerate(extremos):
    for indice2,direcy in enumerate(extremos):

        X1 = rng.random(int(N/2))
        X2 = rng.normal(K-1.5,desv,int(N/2))
        
        X = np.concatenate((X1,X2))
        X[X < 0] = np.zeros(X[X < 0].shape[0])
        X[X > K] = np.ones(X[X > K].shape[0])*K
        
        Y1 = rng.random(int(N/2))
        Y2 = rng.normal(K-1.5,desv,int(N/2))
        
        Y = np.concatenate((Y1,Y2))
        Y[Y < 0] = np.zeros(Y[Y < 0].shape[0])
        Y[Y > K] = np.ones(Y[Y > K].shape[0])*K
        
        X = X*direcx
        Y = Y*direcy
        
        Opiniones = np.zeros(2*N)
        Opiniones[0::2] = X
        Opiniones[1::2] = Y
        
        filename = "../Datos/Opiniones_N=1000_kappa=6_beta=0.6_cosd=0.00_Iter={}.file".format(indice2+2*indice1+12)
        
        Guardar_archivo(Opiniones,filename)
        
        
# Cuarto hago el caso más grande del otro extremo

extremos = np.array([-1,1])
K = 5
desv = 0.25

for indice1,direcx in enumerate(extremos):
    for indice2,direcy in enumerate(extremos):

        X1 = rng.random(int(N/2))
        X2 = rng.normal(1.5,desv,int(N/2))
        
        X = np.concatenate((X1,X2))
        X[X < 0] = np.zeros(X[X < 0].shape[0])
        X[X > K] = np.ones(X[X > K].shape[0])*K
        
        Y1 = rng.random(int(N/2))
        Y2 = rng.normal(1.5,desv,int(N/2))
        
        Y = np.concatenate((Y1,Y2))
        Y[Y < 0] = np.zeros(Y[Y < 0].shape[0])
        Y[Y > K] = np.ones(Y[Y > K].shape[0])*K
        
        X = X*direcx
        Y = Y*direcy
        
        Opiniones = np.zeros(2*N)
        Opiniones[0::2] = X
        Opiniones[1::2] = Y
        
        filename = "../Datos/Opiniones_N=1000_kappa=6_beta=0.6_cosd=0.00_Iter={}.file".format(indice2+2*indice1+16)
        
        Guardar_archivo(Opiniones,filename)
        

##################################################################################
# Armo los gráficos de polarización a dos de los extremos

extremos = np.array([-1,1])
desv = 0.05
K = 5
N = 1000

extremos_tupla = np.array([(x,y) for x in extremos for y in extremos])

indice = -1

for extremo1 in extremos_tupla:
    for extremo2 in extremos_tupla:
        if np.array([x != y for x,y in zip(extremo1,extremo2)]).any():
    
            X1 = rng.normal(K,desv,int(N/2))
            X2 = rng.normal(K,desv,int(N/2))
            
            X1 = X1*extremo1[0]
            X2 = X2*extremo2[0]
            
            X = np.concatenate((X1,X2))
            
            Y1 = rng.normal(K,desv,int(N/2))
            Y2 = rng.normal(K,desv,int(N/2))
            
            Y1 = Y1*extremo1[1]
            Y2 = Y2*extremo2[1]
            
            Y = np.concatenate((Y1,Y2))
            
            Opiniones = np.zeros(2*N)
            Opiniones[0::2] = X
            Opiniones[1::2] = Y
            
            indice = indice + 1
            
            filename = "../Datos/Opiniones_N=1000_kappa=6_beta=1.5_cosd=0.00_Iter={}.file".format(indice)
            
            Guardar_archivo(Opiniones,filename)
            
            

