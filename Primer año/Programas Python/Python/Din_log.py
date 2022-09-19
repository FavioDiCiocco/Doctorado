# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:11:00 2022

@author: Favio
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math
import os
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

T=2 # Defino acá el número de tópicos porque es algo que no cambia por ahora,
# pero no tenía dónde más definirlo


for Carpeta in ["Din_log"]:

   # CÓDIGO PARA LEVANTAR ARCHIVOS DE UNA CARPETA CON TODOS LOS ARCHIVOS MEZCLADOS
    
    CarpCheck=[[root,files] for root,dirs,files in os.walk("../{}".format(Carpeta))]
    
    # El elemento en la posición x[0] es el nombre de la carpeta
    
    for x in CarpCheck:
        # dada = x[0].split("\\")
        Archivos_Datos = [nombre for nombre in x[1]]
        Archivos_Datos.insert(0,x[0])
    
    #-------------------------------------------------------------------------------------------------------
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Opiniones_alfa=$_N=$_Cosd=$_mu=$_Iter=$.file"
    # También tengo otros archivos llamados "Testigos_alfa=$_N=$_Cosd=$_mu=$_Iter=$.file"
    
    Conjunto_Direcciones = Archivos_Datos[0]
    
    # Voy a trabajar mi lista de archivos usando pandas, creo que va a ser mucho más claro lo que
    # está pasando y además va a ser más orgánico.
    Df_archivos = pd.DataFrame({"nombre": Archivos_Datos[1::]})
    
    # Hecho mi dataframe, voy a armar columnas con los parámetros que varían en los nombres de mis archivos
    Df_archivos["tipo"] = Df_archivos["nombre"].apply(lambda x: x.split("_")[0])
    Df_archivos["alfa"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[1].split("=")[1]))
    Df_archivos["n"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[2].split("=")[1]))
    Df_archivos["cdelta"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[3].split("=")[1]))
    Df_archivos["mu"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[4].split("=")[1]))
    Df_archivos["iteracion"] = Df_archivos["nombre"].apply(lambda x: float(x.split("_")[5].split("=")[1].strip(".file")))
    
    # Con esto tengo hecho un DF con los nombres de todos los archivos y además tengo separados los valores
    # de N, alfa, Cdelta y todo lo demás en columnas que me van a servir para filtrar esos archivos. Honestamente
    # creo que es más sencillo armar las columnas ahora y no ir creándolas cada vez que las necesito.
    
    #--------------------------------------------------------------------------------------------
    
    # Armo los gráficos de opinión en función del tiempo a base de los testigos.
    
    func.Graf_opi_vs_tiempo(Df_archivos, Conjunto_Direcciones, Carpeta)
    
    #----------------------------------------------------------------------------------------------
    
    """
    
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    arrayN = np.unique(Df_archivos["n"])
    arrayAlfa = np.unique(Df_archivos["alfa"])
    arrayCdelta = np.unique(Df_archivos["cdelta"])
    arrayMu = np.unique(Df_archivos["mu"])
    
    Tupla_total = [(n,alfa,cdelta,mu) for n in arrayN
                   for alfa in arrayAlfa
                   for cdelta in arrayCdelta
                   for mu in arrayMu]
    
    
    for AGENTES,ALFA,CDELTA,MU in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(Df_archivos.loc[((Df_archivos["tipo"]==TIPO) & 
                                    (Df_archivos["n"]==AGENTES) & 
                                    (Df_archivos["mu"]==MU) & 
                                    (Df_archivos["cdelta"]==CDELTA) & 
                                    (Df_archivos["alfa"]==ALFA), "nombre")])
        
        # Y acá está el truco, si la combinación de N, Alfas, Cdeltas y Mu no es una combinación existente,
        # entonces el pandas estará vacío y listo, no hay archivos sobre los cuales iterar. Supongo que
        # después podré armar esto de forma más particular de ser necesario para los mapas de colores.
        
        #-----------------------------------------------------------------------------------------
    
        """

func.Tiempo(t0)
