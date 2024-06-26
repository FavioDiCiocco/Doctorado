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

T=1 # Defino el número de tópicos
Etapa = Path("Variacion_olvido") # Defino el nombre de la etapa del trabajo en la que estoy

# Defino las carpetas que voy a recorrer. Tiene más sentido definir esto a mano.
Carpetas = ["Datos"]

for carp in Carpetas:
    
    # Transformo estas cosas en paths. Espero que acostumbrarme a esto valga la pena
    Direccion = Path("../{}".format(carp))
    carpeta = Path(carp)
    
    # Recorro las carpetas con datos
    CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]
    
    # Me armo una lista con los nombres de todos los archivos con datos.
    # Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos
    
    Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]
    
    #-------------------------------------------------------------------------------------------------------
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Opiniones_N=$_Lambda=$_Iter=$.file"
    # También tengo otros archivos llamados "Testigos_N=$_Lambda=$_Iter=$.file" y
    # "Saturacion_N=$_Lambda=$_Iter=$.file"

    # Voy a trabajar mi lista de archivos usando pandas, creo que va a ser mucho más claro lo que
    # está pasando y además va a ser más orgánico.
    
    Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})
    
    # Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
    Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
    Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[1].split("=")[1]))
    Df_archivos["lambda"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1].strip(".file")))
    
    # Con esto tengo hecho un DF con los nombres de todos los archivos y además tengo separados los valores
    # de N, lambda e iteración en columnas que me van a servir para filtrar esos archivos. Honestamente
    # creo que es más sencillo armar las columnas ahora y no ir creándolas cada vez que las necesito.
    
    #----------------------------------------------------------------------------------------------
    
    # Lo único que me interesa graficar en este caso son los valores promedios de interés
    # a los que tiende el sistema en función del olvido
    
    func.MaxProm_vs_olvido(Df_archivos, Direccion, Etapa / carpeta)
    

func.Tiempo(t0)