#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:02:54 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

#--------------------------------------------------------------------------------
t0 = time.time()
# Voy a definir TODAS las funciones que voy a usar, total definirlas no roba
# tiempo o proceso al programa.


# Esto printea una cantidad de valores cant de un objeto iterable que paso
# en la parte de lista.
def scan(lista,cant=10):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo():
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))


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


Datos = ldata("../Conjunto_pequeño/Varprom_alfa=0.400_N=3_Cosd=1.000_mu=0.100_Iter=0.file")

# Levanto los datos de Variación Promedio
Var = np.array([float(x) for x in Datos[1][:-1]])

# Esto es el tiempo a graficar
X = np.arange(len(Var))*0.01

plt.rcParams.update({'font.size': 20})
plt.figure("Varprom",figsize = (20, 15))

# Ahora grafico las curvas de Variación de Opiniones
plt.semilogy(X,Var,"--", linewidth = 4)
CritCorte = 0.0005
plt.axhline(CritCorte)
plt.grid(alpha = 0.5)
plt.savefig("VariacionProm.png",bbox_inches = "tight")
plt.close("Varprom")

Tiempo()