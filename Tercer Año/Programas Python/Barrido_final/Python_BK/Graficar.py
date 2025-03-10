#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 16 14:21:12 2022

@author: favio
"""


import pandas as pd
import numpy as np
import time
import os
import funciones as func
from pathlib import Path

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

T=2 # Defino el número de tópicos
Etapa = Path("Barrido_final") # Defino el nombre de la etapa del trabajo en la que estoy

# Defino las carpetas que voy a recorrer. Tiene más sentido definir esto a mano.
Carpetas = ["Beta-Kappa"]

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
    
    # Es importante partir del hecho de que mis archivos llevan por nombre:
    # "Opiniones_N=$_kappa=$_beta=$_Iter=$.file" y "Testigos_N=$_kappa=$_beta=$_Iter=$.file"
    # En la carpeta 1D
    
    # En cambio, en la carpeta 2D llevan por nombre:
    # "Opiniones_N=$_kappa=$_beta=$_cosd=$_Iter=$.file" y "Testigos_N=$_kappa=$_beta=$_cosd=$_Iter=$.file"
    
    Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})
    
    # Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
    Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
    Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[1].split("=")[1]))
    Df_archivos["parametro_x"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["parametro_y"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
    Df_archivos["Extra"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[5].split("=")[1].strip(".file")))
    
    #----------------------------------------------------------------------------------------------

    # Por un lado necesito los nombres que pasaré a los títulos de los archivos
    # ID es por el nombre del parámetro.
    # Todo parámetro que no grafique es un parámetro extra
    
    ID_param_x = "kappa"
    ID_param_y = "beta"
    ID_param_extra_1 = "cosd"
    
    
    # Lo otro que necesito es el nombre que pasaré a los ejes de los gráficos de las funciones
    # SIM significa símbolo, porque esto lo uso para escribir el símbolo de ese parámetro
    # Todo parámetro que no grafique es un parámetro extra
    
    SIM_param_x = r"\kappa"
    SIM_param_y = r"\beta"
    SIM_param_extra_1 = r"cos(\delta)"
    
    # Diccionario con la entropía, Sigma_x, Sigma_y, Promedios y Covarianzas
    # de todas las simulaciones para cada punto del espacio de parámetros.
    Dic_Total = func.Diccionario_metricas(Df_archivos,Direccion, 20, 20)
    
    bines = np.linspace(-3.5,3.5,8)
    
    func.Tiempo(t0)
    
    #----------------------------------------------------------------------------------------------
    
    # Gráficos del espacio de parámetros
    """
    func.Mapas_Colores_1D(Df_archivos, Direccion, Etapa/carpeta, SIM_param_x, SIM_param_y)
    
    
    func.Mapa_Colores_Traza_Covarianza(Df_archivos, Direccion, Etapa/carpeta,
                                          SIM_param_x, SIM_param_y,ID_param_extra_1)
    
    
    func.Mapa_Colores_Entropia_opiniones(Df_archivos, Dic_Total, Direccion, Etapa/carpeta,
                                          SIM_param_x, SIM_param_y,ID_param_extra_1)
    """
    
    func.Mapas_Colores_FEF(Df_archivos, Dic_Total, Direccion, Etapa/carpeta,
                           SIM_param_x, SIM_param_y, ID_param_extra_1)
    
    
    func.Graf_Histograma_opiniones_2D(Df_archivos, Dic_Total, Direccion, Etapa/carpeta, bines, "magma",
                                      ID_param_x, ID_param_y, ID_param_extra_1)
    
#-------------------------------------------------------------------------------------------------------------------------

func.Tiempo(t0)

