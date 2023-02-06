#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:21:12 2022

@author: favio
"""


import pandas as pd
import time
import os
import funciones as func
from pathlib import Path

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

T=2 # Defino el número de tópicos
Etapa = Path("Cambios_parametros") # Defino el nombre de la etapa del trabajo en la que estoy

# Defino las carpetas que voy a recorrer. Tiene más sentido definir esto a mano.
Carpetas = ["2D_Epsilon","2D_Kappa"]
Nombre2 = ["epsilon","kappa"]
Titulo2 = ["\epsilon","\kappa"]

for carp,nombre_parametro_2,titulo_parametro_2 in zip(Carpetas,Nombre2,Titulo2):
    
    # Transformo estas cosas en paths. Espero que acostumbrarme a esto valga la pena
    Direccion = Path("../{}".format(carp))
    carpeta = Path(carp)
    
    # Recorro las carpetas con datos
    CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]
    
    # Me armo una lista con los nombres de todos los archivos con datos.
    # Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos
    
    Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]
    
    #-------------------------------------------------------------------------------------------------------
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Opiniones_N=$_Cosd=$_epsilon=$_Iter=$.file"
    # También tengo otros archivos llamados "Testigos_N=$_Cosd=$_epsilon=$_Iter=$.file" y

    # Voy a trabajar mi lista de archivos usando pandas, creo que va a ser mucho más claro lo que
    # está pasando y además va a ser más orgánico.
    
    Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})
    
    # Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
    Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
    Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[1].split("=")[1]))
    Df_archivos["parametro_1"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["parametro_2"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1].strip(".file")))
    
    # Con esto tengo hecho un DF con los nombres de todos los archivos y además tengo separados los valores
    # de N, lambda e iteración en columnas que me van a servir para filtrar esos archivos. Honestamente
    # creo que es más sencillo armar las columnas ahora y no ir creándolas cada vez que las necesito.
    
    #----------------------------------------------------------------------------------------------
    
    # Modifiqué los archivos de graficación considerando que siempre que grafique lo haré teniendo dos
    # parámetros libres. Entonces la idea es que le pase los nombres de esos parámetros a las funciones
    # para que los pongan en los nombres de los archivos o que los pongan en los gráficos. Fuera de eso
    # el programa trabaja pensando en parámetro 1 y 2, siendo el parámetro 1 el que se grafica sobre
    # el eje Y mientras que el 2 se grafica sobre el eje X, siempre que corresponda.
    
    # Por un lado necesito los nombres que pasaré a los títulos de los archivos
    
    nombre_parametro_1 = "cos(delta)"
    # nombre_parametro_2 = "epsilon"
    
    # Lo otro que necesito es el nombre que pasaré a los ejes de los gráficos de las funciones
    
    titulo_parametro_1 = "cos(\delta)"
    # titulo_parametro_2 = "\epsilon"
    
    #----------------------------------------------------------------------------------------------
    
    # Grafico el promedio en función de la amplitud y el epsilon, armando mi mapa de colores.
    
#    func.Mapa_Colores_Promedio_opiniones(Df_archivos, Direccion, Etapa/carpeta, titulo_parametro_1, titulo_parametro_2)

    #----------------------------------------------------------------------------------------------
    
    # Para tener una mejor idea de lo que estoy viendo, voy a hacer los gráficos de Opi_vs_tiempo
    
#    func.Graf_opi_vs_tiempo(Df_archivos, Direccion, Etapa/carpeta,T, nombre_parametro_1, nombre_parametro_2)
    
    #----------------------------------------------------------------------------------------------
    
    # Para tener una mejor idea de lo que estoy viendo, voy a hacer los gráficos de Opi_vs_tiempo
    
    func.Graf_Punto_fijo_vs_parametro(Df_archivos, Direccion, Etapa/carpeta,T, nombre_parametro_2, titulo_parametro_1, titulo_parametro_2)
    

func.Tiempo(t0)