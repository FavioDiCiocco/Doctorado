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
Etapa = Path("Exploracion_Logistica") # Defino el nombre de la etapa del trabajo en la que estoy

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
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Opiniones_N=$_kappa=$_alfa=$_Iter=$.file"
    # También tengo otros archivos llamados "Testigos_N=$_kappa=$_alfa=$_Iter=$.file" y
    
    Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})
    
    # Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
    Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
    Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[1].split("=")[1]))
    Df_archivos["parametro_1"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["parametro_2"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1].strip(".file")))
    
    #----------------------------------------------------------------------------------------------

    # Por un lado necesito los nombres que pasaré a los títulos de los archivos
    
    nombre_parametro_1 = "kappa"
    nombre_parametro_2 = "alfa"
    
    # Lo otro que necesito es el nombre que pasaré a los ejes de los gráficos de las funciones
    
    titulo_parametro_1 = r"\kappa"
    titulo_parametro_2 = r"\alpha"
    
    #----------------------------------------------------------------------------------------------
    
    func.Mapa_Colores_Promedio_opiniones(Df_archivos, Direccion, Etapa/carpeta,
                                         titulo_parametro_1, titulo_parametro_2, True)

    #----------------------------------------------------------------------------------------------
    
#    func.Graf_opi_vs_tiempo(Df_archivos, Direccion, Etapa/carpeta,T, nombre_parametro_1, nombre_parametro_2)
    
    #----------------------------------------------------------------------------------------------
    """
    Cosdeltas = [0,0.2,0.4,0.6,0.8,1]
    
    for CDELTA in Cosdeltas:
        
        # Fijo el parámetro de Cosdelta y barro en función de epsilon
        Df_Cosdelta_fijo = (Df_archivos.query('parametro_1 == @CDELTA')
                            .rename(columns={"parametro_1":"cosdelta",
                                              "parametro_2":"parametro_1",
                                              "parametro_3":"parametro_2"})
                            .copy()
                        )
        
        func.Graf_Punto_fijo_vs_parametro(Df_Cosdelta_fijo, Direccion,
                                          Etapa/carpeta,T, nombre_parametro_3,
                                          titulo_parametro_2, titulo_parametro_3,
                                          Condicion_punto_inestable_Kappa_Epsilon=True)
        
        
        
        func.Mapa_Colores_Varianza_opiniones(Df_Cosdelta_fijo, Direccion,
                                             Etapa/carpeta,
                                             titulo_parametro_2, titulo_parametro_3,
                                             True)
        
        #----------------------------------------------------------------------------------------------
        
        # Ahora fijo el parámetro de Cosdelta y barro en función de Kappa
        
        # Fijo el parámetro de Cosdelta y barro en función de epsilon
        Df_Cosdelta_fijo = (Df_archivos.query('parametro_1 == @CDELTA')
                            .rename(columns={"parametro_1":"cosdelta",
                                              "parametro_3":"parametro_1"})
                            .copy()
                        )
        
        func.Graf_Punto_fijo_vs_parametro(Df_Cosdelta_fijo, Direccion,
                                          Etapa/carpeta,T, nombre_parametro_2,
                                          titulo_parametro_3, titulo_parametro_2,
                                          Condicion_punto_inestable_Epsilon_Kappa=True)
        
        #----------------------------------------------------------------------------------------------
        
    
    # Armo el gráfico 3D de los puntos fijos de interés en función de los parámetros Kappa y Epsilon
    
    func.Graf_Punto_fijo_3D(Df_archivos, Direccion, Etapa/carpeta,T,
                            titulo_parametro_1, titulo_parametro_2, titulo_parametro_3)
    """


func.Tiempo(t0)