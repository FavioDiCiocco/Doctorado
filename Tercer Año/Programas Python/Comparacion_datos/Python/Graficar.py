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
Etapa = Path("Comparacion_datos") # Defino el nombre de la etapa del trabajo en la que estoy

# Defino las carpetas que voy a recorrer. Tiene más sentido definir esto a mano.
Carpetas = ["Zoom_Beta-Cosd"]

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
    
    func.Tiempo(t0)
    
    #----------------------------------------------------------------------------------------------
    
    bines = np.linspace(-3.5,3.5,8)
    
    Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)
    """
    labels_politicos = ['V201372x','V201386x','V201426x']

    labels_apoliticos = ['V201429','V202320x','V202341x','V202350x']

    labels_dudosos = ['V201225x','V201262','V202242x','V202248x']
    
    # Esta parte del código la uso para calcular el mapa de colores de DJS de varios pares de preguntas,
    # todo de un plumazo.
    
    labels = [('V201372x','V201386x','V200010a'), ('V201426x','V201386x','V200010a'), ('V202341x','V202331x','V200010b')] #,
              # ('V202350x','V202341x','V200010b'),('V201262','V202248x','V200010b'),('V202242x','V202248x','V200010b')]
    
    rangos = [(np.array([0,0.1]),np.array([0.4,0.8])), (np.array([0,0.15]),np.array([0.5,0.7])), (np.array([0,0.1]),np.array([0.4,0.7]))] #,
              # (np.array([0,0.2]),np.array([0.4,0.8])), (np.array([0,0.1]),np.array([0.4,0.66])), (np.array([0,0.1]),np.array([0.4,0.7]))]
    
    for preguntas,rango_ajuste in zip(labels,rangos):
            
        code_1 = preguntas[0]
        code_2 = preguntas[1]
        weights = preguntas[2]
        
        Dic_ANES = {"code_1": code_1, "code_2": code_2, "weights":weights}
        
        func.Mapas_Colores_DJS(Df_archivos, Df_ANES, dict_labels, Direccion, Etapa/carpeta, Dic_ANES,bines,
                              ID_param_x, SIM_param_x, ID_param_y, SIM_param_y)
        
        #----------------------------------------------------------------------------------------------
        
        # Esta parte del código la uso para calcular los parámetros del ajuste paraboloidico aplicado
        # a los datos de las distancias.
        
        x_range = rango_ajuste[0]
        y_range = rango_ajuste[1]
        
        params_centro, params_cruz = func.Ajuste_DJS(Df_archivos, Df_ANES, Direccion, Etapa/carpeta, Dic_ANES,
                                 x_range,y_range)
    
        # Define the mathematical function
        def my_function(x, y, params):
            return params[0]*y**2 + params[1]*y + params[2]*x**2 + params[3]*x + params[4]
        
        func.plot_3d_surface(Etapa/carpeta, Dic_ANES, my_function, params_centro, params_cruz, x_range,
                             y_range,SIM_param_x, SIM_param_y)
        
        
    #    func.plot_3d_scatter(Df_archivos, Df_ANES, Direccion, Etapa/carpeta, Dic_ANES,
    #                         np.array([0.5,0.72]),np.array([0.04,0.15]), SIM_param_x, SIM_param_y)
        
        
        initial_guess = [0.1,0.5]
        
        def my_function_minimize(x, a,b,c,d,e):
            return a*x[1]**2 + b*x[1] + c*x[0]**2 + d*x[0] + e
        
        # Perform the minimization
        result_centro = minimize(my_function_minimize, initial_guess, args=tuple(params_centro))
        result_cruz = minimize(my_function_minimize, initial_guess, args=tuple(params_cruz))
        
        # Print the result
        print("Variables ajustadas para preguntas: {} vs {}".format(dict_labels[code_2],dict_labels[code_1]))
        print("Variables óptimas distribución sin centro:", result_centro.x)
        print("Variables óptimas distribución sin cruz:", result_cruz.x)
    
        """
#    bines = np.linspace(-3.5,3.5,8)
#    func.Graf_Histograma_opiniones_2D(Df_archivos, Direccion, Etapa/"distribuciones", bines, "magma",
#                                      ID_param_x, ID_param_y, ID_param_extra_1)
        
# Este código fue una prueba de cosas para ver cómo calcular la distancia
# Jensen-Shannon. Ahora lo aislo porque ya generalicé esto en una función.

# Consideremos que quiero revisar los estados finales de las simulaciones contra uno de mis
    # gráficos. Tengo que tomar las opiniones finales de algún estado y armar la distribución
    # asociada.
    
    Sim_prueba = Df_archivos.loc[(Df_archivos["tipo"]=="Opiniones") & 
                                (Df_archivos["n"]==1000) & 
                                (Df_archivos["Extra"]==10) & 
                                (Df_archivos["parametro_x"]==0) &
                                (Df_archivos["parametro_y"]==0.5) & 
                                (Df_archivos["iteracion"]==1), "nombre"]
    
    for nombre in Sim_prueba:
        Datos = func.ldata(Direccion / nombre)
    
    # Recordemos que los archivos tienen forma
    
    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
    # Opinión Inicial del sistema
    # Variación Promedio
    # Opinión Final
    # Pasos simulados
    # Semilla
    # Matriz de Adyacencia
    
    Opifinales = np.array(Datos[5][:-1], dtype="float")
    Opifinales = Opifinales / 10
    
    Distr_Sim = np.reshape(func.Clasificacion(Opifinales,7,7,2),(49,1))
    
    # Rearmo la distribución de la simulación de forma de que haya un agente
    # en cada punto como mínimo. La idea es no tener ceros en ningún lugar.
    # Después resto todos los agentes que sumé del lugar que tenga más agentes
    restar = np.count_nonzero(Distr_Sim == 0)
    ubic = np.argmax(Distr_Sim)
    Distr_Sim[Distr_Sim == 0] = np.ones(restar)*0.001
    Distr_Sim[ubic] = Distr_Sim[ubic] - 0.001*restar
    
    
    # Ya tengo la distribución de opiniones de mi simulación, ahora necesito la
    # de la encuesta ANES.
    
    code_1 = 'V201200' 
    code_2 = 'V201420x'
    weights = 'V200010a'
    
    # Extraigo la distribución en hist2d
    df_aux = Df_ANES.loc[(Df_ANES[code_1]>0) & (Df_ANES[code_2]>0)]
    # hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_1], y=df_aux[code_2], weights=df_aux[weights], vmin=0,cmap = "inferno",
    #           bins=[np.arange(df_aux[code_1].min()-0.5, df_aux[code_1].max()+1.5, 1), np.arange(df_aux[code_2].min()-0.5, df_aux[code_2].max()+1.5, 1)])
    
    # Filter out rows where either code_1 or code_2 is 3
    df_filtered = df_aux[(df_aux[code_1] != 4) | (df_aux[code_2] != 4)] # Sólo saca el centro
    # df_filtered = df_aux[(df_aux[code_1] != 4) & (df_aux[code_2] != 4)] # Saca la cruz
    hist2d_r, xedges, yedges, im = plt.hist2d(x=df_filtered[code_1], y=df_filtered[code_2], weights=df_filtered[weights], vmin=0,cmap = "inferno",
              bins=[np.arange(df_filtered[code_1].min()-0.5, df_filtered[code_1].max()+1.5, 1), np.arange(df_filtered[code_2].min()-0.5, df_filtered[code_2].max()+1.5, 1)])
    
    plt.colorbar()
    Distr_Enc = np.reshape(hist2d_r,(hist2d_r.shape[0]*hist2d_r.shape[1],1))
    
    # Para comparar las distribuciones, les remuevo el elemento del centro.
    # Quizás la función de Jensen Shannon está quitando los elementos con cero
    # por sí misma, pero ni idea.
    
    Distr_Sim = np.delete(Distr_Sim,24)
    Distr_Enc = np.delete(Distr_Enc,24)
    
    # Una vez que tengo las dos distribuciones, hago el cálculo de la distancia
    # Jensen-Shannon
    
    distancia = jensenshannon(Distr_Enc,Distr_Sim)
    print(distancia)
    
    # Esto que me armé efectivamente calcula la distancia Jensen-Shannon entre dos
    # distribuciones. Extrañamente, no tiene problemas con las distribuciones que tengan
    # ceros. Muy raro.

        
        #----------------------------------------------------------------------------------------------

func.Tiempo(t0)

#########################################################################################
#########################################################################################

