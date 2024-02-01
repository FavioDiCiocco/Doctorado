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
Etapa = Path("Interes_actualizado") # Defino el nombre de la etapa del trabajo en la que estoy

# Defino las carpetas que voy a recorrer. Tiene más sentido definir esto a mano.
Carpetas = ["Datos/gm=6"]

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
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Opiniones_N=$_Cosd=$_kappa=$_epsilon=$_Iter=$.file"
    # También tengo otros archivos llamados "Testigos_N=$_Cosd=$_kappa=$_epsilon=$_Iter=$.file" y
    
    Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})
    
    # Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
    Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
    Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: int(x.split("_")[1].split("=")[1]))
    Df_archivos["Extra"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["parametro_x"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1]))
    Df_archivos["parametro_y"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[5].split("=")[1].strip(".file")))
    
    #----------------------------------------------------------------------------------------------

    # Por un lado necesito los nombres que pasaré a los títulos de los archivos
    # ID es por el nombre del parámetro.
    # Todo parámetro que no grafique es un parámetro extra
    
    ID_param_extra_1 = "Cosd"
    ID_param_x = "epsilon"
    ID_param_y = "kappa"
    
    # Lo otro que necesito es el nombre que pasaré a los ejes de los gráficos de las funciones
    # SIM significa símbolo, porque esto lo uso para escribir el símbolo de ese parámetro
    # Todo parámetro que no grafique es un parámetro extra
    
    SIM_param_extra_1 = r"Cos(\delta)"
    SIM_param_x = r"\epsilon"
    SIM_param_y = r"\kappa"
    
    #----------------------------------------------------------------------------------------------
    
    # func.Mapa_Colores_Promedio_opiniones(Df_archivos, Direccion, Etapa/carpeta,
    #                                      SIM_param_x, SIM_param_y, True)

    #----------------------------------------------------------------------------------------------
    
    # func.Mapa_Colores_Tiempo_convergencia(Df_archivos, Direccion, Etapa/carpeta,
    #                                      SIM_param_x, SIM_param_y, True)
    
    #----------------------------------------------------------------------------------------------
    
    func.Graf_opi_vs_tiempo(Df_archivos, Direccion, Etapa/carpeta, T,
                            ID_param_x = "epsilon", ID_param_y = "kappa")
    
    #----------------------------------------------------------------------------------------------
    
    # func.Graf_Derivada_vs_tiempo(Df_archivos, Direccion, Etapa/carpeta, T,
    #                         nombre_parametro_1, nombre_parametro_2)


func.Tiempo(t0)