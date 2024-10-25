#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 16 14:21:12 2022

@author: favio

"""


import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.optimize import minimize
import matplotlib.pyplot as plt
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
Carpetas = ["Beta-Cosd"]

for carp in Carpetas:
    
    # Transformo estas cosas en paths
    Direccion = Path("../{}".format(carp))
    carpeta = Path(carp)
    # Cambio la carpeta para el código de los clusters
    # carpeta = Path("B04C00Cluster")
    
    Dir_matrices_JS = Path("../Matrices DJS")
    Dir_matrices_KS = Path("../Matrices DKS")
    
    # Recorro las carpetas con archivos csv
    CarpMatJS=[[root,files] for root,dirs,files in os.walk(Dir_matrices_JS)]
    CarpMatKS=[[root,files] for root,dirs,files in os.walk(Dir_matrices_KS)]
    
    # Recorro las carpetas con datos
    CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]
    
    # Me armo una lista con los nombres de todos los archivos con datos
    Archivos_Datos = CarpCheck[0][1]
    # Me armo una lista con los nombres de todos los archivos csv
    Archivos_Matrices_JS = CarpMatJS[0][1]
    Archivos_Matrices_KS = CarpMatKS[0][1]
    
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
    Df_archivos["Extra"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["parametro_y"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
    Df_archivos["parametro_x"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[5].split("=")[1].strip(".file")))
    
    #----------------------------------------------------------------------------------------------
    
    # Por un lado necesito los nombres que pasaré a los títulos de los archivos
    # ID es por el nombre del parámetro.
    # Todo parámetro que no grafique es un parámetro extra
    
    ID_param_extra_1 = "kappa"
    ID_param_y = "beta"
    ID_param_x = "cosd"
    
    # Lo otro que necesito es el nombre que pasaré a los ejes de los gráficos de las funciones
    # SIM significa símbolo, porque esto lo uso para escribir el símbolo de ese parámetro
    # Todo parámetro que no grafique es un parámetro extra
    
    SIM_param_extra_1 = r"\kappa"
    SIM_param_y = r"\beta"
    SIM_param_x = r"cos(\delta)"
    
    # Diccionario con la entropía, Sigma_x, Sigma_y, Promedios y Covarianzas
    # de todas las simulaciones para cada punto del espacio de parámetros.
    Dic_Total = func.Diccionario_metricas(Df_archivos,Direccion, 20, 20)
    
    size_x = np.unique(Df_archivos["parametro_x"]).shape[0]
    size_y = np.unique(Df_archivos["parametro_y"]).shape[0]
    
    bines = np.linspace(-3.5,3.5,8)
    
    Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)
    
    """
    # Armo el pandas con la data de las preguntas en un cluster
    # Cluster a revisar: Beta=0.4, Cosd = 0
    
    Beta = 0.4
    Cosd = 0
    Df_cluster = Df_preguntas.loc[(Df_preguntas["Beta_100"]==Beta) & (Df_preguntas["Cosd_100"]==Cosd)]
    """
    
    func.Tiempo(t0)
    
    
    #----------------------------------------------------------------------------------------------
    
    # Gráficos del espacio de parámetros
    
    func.Mapa_Colores_Entropia_opiniones(Df_archivos, Dic_Total, Direccion, Etapa/carpeta,
                                         SIM_param_x, SIM_param_y,ID_param_extra_1)
    
    func.Mapa_Colores_Covarianzas(Df_archivos, Direccion, Etapa/carpeta,
                                  SIM_param_x, SIM_param_y, ID_param_extra_1)
    
    func.Mapas_Colores_FEF(Df_archivos, Dic_Total, Direccion, Etapa/carpeta,
                           SIM_param_x, SIM_param_y, ID_param_extra_1)
    
    # lminimos = [(0,0.4),(0,0.6),(0.02,0.5)]
    
    func.Graf_Histograma_opiniones_2D(Df_archivos, Dic_Total, Direccion, Etapa/carpeta, bines, "magma",
                                      ID_param_x, ID_param_y, ID_param_extra_1)
    
    
    #----------------------------------------------------------------------------------------------
    
    # Construyo un diccionario con las preguntas de cada uno de los clusters
    
    Df_preguntas = func.Tabla_datos_preguntas(Df_archivos, dict_labels, Archivos_Matrices_JS, Dir_matrices_JS)
    # Df_preguntas.to_csv("Tabla_JS.csv", index=False)
    
    # Lo inicializo en 6 para que los no revisados sean un cluster aparte
    Df_preguntas["clusters"] = 1
    # Clusters = [(0,0.4), (0,0.6), (0.02,1.1), (0.08,1.1), (0.14,1.1), (0.48,0.4)]
    
    # for cluster,tupla in enumerate(Clusters):
    #     Cosd = tupla[0]
    #     Beta = tupla[1]
        
    #     Df_preguntas.loc[(Df_preguntas["Cosd_100"]==Cosd) & (Df_preguntas["Beta_100"]==Beta), "clusters"] = cluster
    
    # Voy a armar gráficos con menos puntos en el espacio de parámetros
    # Estos son los estados Radicalizados
    # Lista_grafs = ["V202331x_vs_V201372x.csv","V202331x_vs_V202336x.csv","V202336x_vs_V201372x.csv",
    #                "V202341x_vs_V201372x.csv","V202341x_vs_V202331x.csv","V202341x_vs_V202336x.csv",
    #                "V202350x_vs_V201372x.csv","V202350x_vs_V202331x.csv","V202350x_vs_V202336x.csv",
    #                "V202350x_vs_V202341x.csv","V202383x_vs_V201372x.csv","V202383x_vs_V202331x.csv",
    #                "V202383x_vs_V202336x.csv","V202383x_vs_V202341x.csv","V202383x_vs_V202350x.csv"]
    
    # Este es un subconjunto de distribuciones que parecen ser los más parecidos a estados finales
    # de la simulación.
    Lista_subconj = ["V201411xvsV201408x.csv","V201426xvsV201386x.csv","V202255xvsV201420x.csv",
                     "V202255xvsV202328x.csv","V202255xvsV202341x.csv","V202328xvsV201386x.csv",
                     "V202331xvsV201386x.csv","V202341xvsV201372x.csv","V202341xvsV201386x.csv",
                     "V202341xvsV202331x.csv","V202350xvsV201386x.csv","V202350xvsV202341x.csv"]
    
    Df_preguntas = Df_preguntas[Df_preguntas["nombre"].isin(Lista_subconj)]
    
    #----------------------------------------------------------------------------------------------
    
    # Gráficos de métricas JS
    
    func.Preguntas_espacio_parametros(Df_archivos, Df_preguntas, Dir_matrices_JS, Etapa/carpeta, "JS",
                                      SIM_param_x, SIM_param_y)
    
    # Df_preguntas = func.Tabla_datos_preguntas(Df_archivos, dict_labels, Archivos_Matrices_JS, Dir_matrices_JS)
    # Df_preguntas.to_csv("Tabla_JS.csv", index=False)
    
    for nombre_csv in Archivos_Matrices_JS:
        
        DJS, code_x, code_y = func.Lectura_csv_Matriz(size_y, size_x, Dir_matrices_JS, nombre_csv)
        
        func.Mapas_Colores_csv(DJS, code_x, code_y, Df_archivos, dict_labels, "JS", Etapa/carpeta,
                               ID_param_x,SIM_param_x,ID_param_y,SIM_param_y)
        
        func.Hist2D_similares_FEF(DJS, code_x, code_y, Df_archivos, Dic_Total, dict_labels, Etapa/carpeta, Direccion, bines,
                                  "JS",SIM_param_x,SIM_param_y)
    
    
    #----------------------------------------------------------------------------------------------
    
    # Gráficos de métricas KS
    
    Df_preguntas_KS = func.Tabla_datos_preguntas(Df_archivos, dict_labels, Archivos_Matrices_KS, Dir_matrices_KS)
    Df_preguntas_KS["clusters"] = Df_preguntas["clusters"]
    # Df_preguntas.to_csv("Tabla_KS.csv", index=False)
    
    Df_preguntas_KS = Df_preguntas_KS[Df_preguntas_KS["nombre"].isin(Lista_subconj)]
    
    func.Preguntas_espacio_parametros(Df_archivos, Df_preguntas_KS, Dir_matrices_KS, Etapa/carpeta, "KS",
                                      SIM_param_x, SIM_param_y)
    
    for nombre_csv in Archivos_Matrices_KS:
        
        DKS, code_x, code_y = func.Lectura_csv_Matriz(size_y, size_x, Dir_matrices_KS, nombre_csv)
        
        func.Mapas_Colores_csv(DKS, code_x, code_y, Df_archivos, dict_labels, "KS", Etapa/carpeta,
                               ID_param_x,SIM_param_x,ID_param_y,SIM_param_y)
        
        func.Hist2D_similares_FEF(DKS, code_x, code_y, Df_archivos, Dic_Total, dict_labels, Etapa/carpeta, Direccion, bines,
                                  "KS",SIM_param_x,SIM_param_y)
    
    #----------------------------------------------------------------------------------------------
    
func.Tiempo(t0)
