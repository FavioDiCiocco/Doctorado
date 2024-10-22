#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pathlib import Path
import os
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

#####################################################################################

# Esta es la función que uso por excelencia para levantar datos de archivos. Lo
# bueno es que lee archivos de forma general, no necesita que sean csv o cosas así
def ldata(archive):
    with open(archive) as f:
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data

#####################################################################################

"""

# Voy a ver de levantar datos de los archivos nuevos

# Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
# Distribución final
# Semilla
# Fragmentos Matriz de Adyacencia

# Levanto los datos del archivo
Datos = ldata("../Beta-Cosd/Opiniones_N=10000_kappa=10_beta=0.40_cosd=0.00_Iter=0.file")

puntos_medios = (np.linspace(-1,1,43)[0:-1] + np.linspace(-1,1,43)[1:])/2

dist_simul = np.reshape(np.array(Datos[1],dtype="float"),(42,42))

Opiniones = np.zeros(2*10000)
agregados = 0

for fila in range(dist_simul.shape[0]):
    for columna in range(dist_simul.shape[1]):
        
        cant_agregar = round(dist_simul[fila,columna] * 10000)
        if (cant_agregar > 0):
            x_i = puntos_medios[fila]
            y_i = puntos_medios[columna]
            
            vector_agregar = np.zeros(cant_agregar*2)
            vector_agregar[0::2] = np.ones(cant_agregar)*x_i
            vector_agregar[1::2] = np.ones(cant_agregar)*y_i
            
            Opiniones[agregados*2:(cant_agregar+agregados)*2] = vector_agregar
            
            agregados += cant_agregar
        
print(agregados)



#####################################################################################

# Recorro las carpetas con datos
CarpCheck=[[root,files] for root,dirs,files in os.walk("../Beta-Cosd")]

# Me armo una lista con los nombres de todos los archivos con datos.
# Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos

Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]

for nombre in Archivos_Datos:
    
    file_path = Path("../Beta-Cosd/{}".format(nombre))
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Modify the second line
    lines[1] = lines[1].rstrip('\n') + '\t\n'
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

"""

#####################################################################################

# Voy a armar la función que construya los archivos csv con las matrices de distancia
# Jensen-Shannon. La idea es que cada fila del csv tenga uno de los elementos de mi
# matriz, recordando que la matriz es de NxMxP elementos. Entonces cada fila del
# csv debería tener P elementos.

#-------------------------------------------------------------------------------------
"""
# Primero me armo el data frame con los nombres de los archivos

# Recorro las carpetas con datos
CarpCheck=[[root,files] for root,dirs,files in os.walk("../Beta-Cosd")]

# Me armo una lista con los nombres de todos los archivos con datos.
# Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos

Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]

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

#-------------------------------------------------------------------------------------

# Segundo construyo el conjunto de preguntas al que quiero calcularles las matrices 

# Gráficos de las preguntas ANES

bines = np.linspace(-3.5,3.5,8)

Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)

labels_politicos = ['V201372x','V201386x','V201408x','V201411x','V201420x','V201426x',
                    'V202255x','V202328x','V202336x']

labels_apoliticos = ['V201429','V202320x','V202331x','V202341x','V202344x','V202350x','V202383x']

labels = []

for i,code_1 in enumerate(labels_politicos):
    for code_2 in labels_politicos[i+1:]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
            
        labels.append((code_1,code_2,weights))
    
    
for i,code_1 in enumerate(labels_apoliticos):
    for code_2 in labels_apoliticos[i+1:]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
            
        labels.append((code_1,code_2,weights))
    
    
for code_1 in labels_politicos:
    for code_2 in labels_apoliticos:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
            
        labels.append((code_1,code_2,weights))


#-------------------------------------------------------------------------------------

# Tercero calculo la matriz de distancia Jensen-Shannon y guardo eso en un csv.


for preguntas in labels[::10]:
    
    code_1 = preguntas[0]
    code_2 = preguntas[1]
        
    weights = preguntas[2]
    
    Dic_ANES = {"code_1": code_1, "code_2": code_2, "weights":weights}
    
    DJS, code_x, code_y = func.Matriz_DJS(Df_archivos, Df_ANES, Dic_ANES, Path("../Beta-Cosd"))
    
    DJS_alterado = np.reshape(DJS, (DJS.shape[0]*DJS.shape[1],DJS.shape[2]))
    
    np.savetxt("../Matrices DJS/{}_vs_{}.csv".format(code_y,code_x), DJS_alterado,delimiter = ",", fmt = "%.4f")


#-------------------------------------------------------------------------------------

# Cuarto armo un código que levante los datos del archivo csv y de ahí
# reconstruya la matriz de DJS

Arr_param_x = np.unique(Df_archivos["parametro_x"])
Arr_param_y = np.unique(Df_archivos["parametro_y"])

# Reviso los archivos en la carpeta de las matrices

CarpMatrices=[[root,files] for root,dirs,files in os.walk("../Matrices DJS")]

# Me armo una lista con los nombres de todos los archivos con datos.
# Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos

Archivos_Matrices = [nombre for nombre in CarpMatrices[0][1]]

# De esos archivos, tomo los nombres de las preguntas que voy a considerar.

for nombre in Archivos_Matrices:
    code_y = nombre.split("_")[0]
    code_x = nombre.split("_")[2]
    
    mat_archivo = np.loadtxt("../Matrices DJS/{}".format(nombre), delimiter = ",")
    DJS = np.reshape(mat_archivo, (Arr_param_y.shape[0],Arr_param_x.shape[0],mat_archivo.shape[1]))
    
    print(nombre)

"""

#####################################################################################
#####################################################################################

# Armo el gráfico de las regiones del espacio de parámetros Beta-Cosd


"""

tlinea = 6


# Create a figure and axis
plt.rcParams.update({'font.size': 44})
fig, ax = plt.subplots(figsize=(28,21))

# Región de Polarización Descorrelacionada
x = [0, 0.1, 0.15, 0, 0]  # x-coordinates
y = [1.1, 1.1, 1.5, 1.5, 1.1]  # y-coordinates
# ax.fill(x, y, color='tab:orange')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) #, label='Polarización Descorrelacionada')
ax.text(0.05, 1.3, 'I', fontsize=40, ha='center', va='center', color='k')

# Región de Transición
x = [0.1, 0.15, 0.3, 0.15, 0.1]  # x-coordinates
y = [1.1, 1.1, 1.5, 1.5, 1.1]  # y-coordinates
# ax.fill(x, y, color='tab:olive')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)
ax.text(0.18, 1.3, 'II', fontsize=40, ha='center', va='center', color='k')

# Región de Polarización ideológica
x = [0.15, 0.5, 0.5, 0.3, 0.15] # x-coordinates
y = [1.1, 1.1, 1.5, 1.5, 1.1] # y-coordinates
# ax.fill(x, y, color='tab:red')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)
ax.text(0.35, 1.3, 'III', fontsize=40, ha='center', va='center', color='k')

# Región de Consenso Radicalizado
x = [0, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.1, 0.1, 0, 0]  # x-coordinates
y = [0, 0, 0.15, 0.3, 0.6, 0.6, 1.1, 1.1, 0.3, 0.2, 0]  # y-coordinates
# ax.fill(x, y, color='tab:green')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)
ax.text(0.3, 0.85, 'VI', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 1
x = [0, 0.1, 0.1, 0, 0] # x-coordinates
y = [0.2, 0.3, 0.75, 0.75, 0.2] # y-coordinates
# ax.fill(x, y, color='tab:purple')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea)  # label=r'Mezcla: CR (50~80%), P1Da (20~35%)')
ax.text(0.05, 0.55, 'V', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 2
x = [0, 0.1, 0.1, 0, 0] # x-coordinates
y = [0.75, 0.75, 1.1, 1.1, 0.75] # y-coordinates
# ax.fill(x, y, color='tab:brown')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (30~40%), P1D (10~50%)')
ax.text(0.05, 0.9, 'IV', fontsize=40, ha='center', va='center', color='k')

# Región de Mezcla 3
x = [0.2, 0.5, 0.5, 0.2, 0.2] # x-coordinates
y = [0.3, 0.15, 0.6, 0.6, 0.3] # y-coordinates
# ax.fill(x, y, color='tab:blue')  # 'alpha' adjusts transparency
ax.plot(x, y, color='k', linewidth=tlinea) # label=r'Mezcla: CR (40~80%) y PIa (10~45%)')
ax.text(0.3, 0.45, 'VII', fontsize=40, ha='center', va='center', color='k')


ax.set_xlabel(r"$cos(\delta)$")
ax.set_ylabel(r"$\beta$")
ax.set_title("Distribución de estados en el espacio de parámetros")
ax.set_xlim(0,0.5)
ax.set_ylim(0,1.5)
# ax.legend()

direccion_guardado = Path("../../../Imagenes/Barrido_final/Beta-Cosd/Distribucion de estados.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()
"""

#####################################################################################
#####################################################################################
"""
# Voy a intentar armar una función que levante datos de algún conjunto de
# datos y calcule la distancia de Kolmogorov-Smirnoff 2D

# Primero levanto los datos de la ANES
Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)
# tuplas_preguntas = [('V201372x','V201386x','V200010a'), ('V201408x','V201426x','V200010a'), ('V201372x','V201411x','V200010a')]

labels_politicos = ['V201372x','V201386x','V201408x','V201411x','V201420x','V201426x',
                    'V202255x','V202328x','V202336x']

labels_apoliticos = ['V201429','V202320x','V202331x','V202341x','V202344x','V202350x','V202383x']

labels = []


for i,code_1 in enumerate(labels_politicos):
    for code_2 in labels_politicos[i+1:]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
            
        labels.append((code_1,code_2,weights))
    
    
for i,code_1 in enumerate(labels_apoliticos):
    for code_2 in labels_apoliticos[i+1:]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
            
        labels.append((code_1,code_2,weights))
    
    
for code_1 in labels_politicos:
    for code_2 in labels_apoliticos:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
            
        labels.append((code_1,code_2,weights))


for preguntas in labels[::5]:
    
    # Separo las opiniones de 0
    df_aux = Df_ANES.loc[(Df_ANES[preguntas[0]]>0) & (Df_ANES[preguntas[1]]>0)]
    # Reviso la cantidad de respuestas de cada pregunta
    resp_1 = np.unique(df_aux[preguntas[0]]).shape[0]
    resp_2 = np.unique(df_aux[preguntas[1]]).shape[0]
    
    # Los clasifico como código x y código y
    if resp_1 >= resp_2:
        code_x = preguntas[0]
        code_y = preguntas[1]
    else:
        code_x = preguntas[1]
        code_y = preguntas[0]
    
    # Dos preguntas con siete respuestas
    if resp_1 == 7 and resp_2 == 7:
        
        # Saco la cruz
        df_filtered = df_aux[(df_aux[code_x] != 4) & (df_aux[code_y] != 4)] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[preguntas[2]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:][:,np.arange(7) != 3]
    
    # Una pregunta con siete respuestas y otra con seis
    elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
        
        # Saco la cruz
        df_filtered = df_aux[df_aux[code_x] != 4] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[preguntas[2]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:]
    
    # Dos preguntas con seis respuestas
    elif resp_1 == 6 and resp_2 == 6:
        
        # No hay necesidad de sacar la cruz
        hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_x], y=df_aux[code_y], weights=df_aux[preguntas[2]], vmin=0,cmap = "inferno", density = True,
                  bins=[np.arange(df_aux[code_x].min()-0.5, df_aux[code_x].max()+1.5, 1), np.arange(df_aux[code_y].min()-0.5, df_aux[code_y].max()+1.5, 1)])
        Distr_Enc = hist2d
        plt.close()
    
    
    ###############################################################################################################
    
    # Lo siguiente que hago es levantar los datos de un archivo
    
    bines = np.linspace(-3.5,3.5,8)
    
    # Recorro las carpetas con datos
    path = Path("../Beta-Cosd")
    CarpCheck=[[root,files] for root,dirs,files in os.walk(path)]
    # Me armo una lista con los nombres de todos los archivos con datos
    Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]
    
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
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(Df_archivos["n"]))
    frac_agente_ind = 1/AGENTES
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(Df_archivos["Extra"]))
    Arr_param_x = np.unique(Df_archivos["parametro_x"])
    Arr_param_y = np.unique(Df_archivos["parametro_y"])
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((XX.shape[0],XX.shape[1],100))
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    T = 2
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(Df_archivos.loc[(Df_archivos["tipo"]==TIPO) & 
                                    (Df_archivos["n"]==AGENTES) & 
                                    (Df_archivos["Extra"]==EXTRAS) & 
                                    (Df_archivos["parametro_x"]==PARAM_X) &
                                    (Df_archivos["parametro_y"]==PARAM_Y), "nombre"])
        
        #-----------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Distribución final
            # Semilla
            # Fragmentos Matriz de Adyacencia
            
            # Levanto los datos del archivo
            
            Datos = ldata(path / nombre)
    
            dist_final = np.reshape(np.array(Datos[1][:-1],dtype="float"),(42,42))
            Opifinales = func.Reconstruccion_opiniones(dist_final, AGENTES, T)
            
            Distr_Sim = func.Clasificacion(Opifinales,hist2d.shape[0],hist2d.shape[1],T)
            Distr_Sim = np.reshape(Distr_Sim, hist2d.shape)
            
            # Dos preguntas con siete respuestas
            if resp_1 == 7 and resp_2 == 7:
                Distr_Sim = Distr_Sim[np.arange(7) != 3,:][:,np.arange(7) != 3]
            
            # Una pregunta con siete respuestas y otra con seis
            elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
                Distr_Sim = Distr_Sim[np.arange(7) != 3,:]
            
            # Como removí parte de mi distribución, posiblemente ya no esté normalizada
            # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
            # de que otra vez esté normalizada
            Distr_Sim = Distr_Sim.flatten()
            if np.sum(Distr_Sim) != 1:
                agentes_agregar = int((1-np.sum(Distr_Sim))/frac_agente_ind)
                for i in range(agentes_agregar):
                    Distr_Sim[np.argmin(Distr_Sim)] += frac_agente_ind
            Distr_Sim = np.reshape(Distr_Sim, (6,6))
            
            Mat_Dist = np.zeros((4,6,6))
            for rotacion in range(4):
                
                Distr_Sim = func.Rotar_matriz(Distr_Sim)
                
                for F in range(6):
                    for C in range(6):
                        
                        DKS = np.zeros(4)
                        
                        # Calculo la distancia en el primer cuadrante (Derecha-arriba)
                        DKS[0] = np.sum(Distr_Enc[F:,C:])-np.sum(Distr_Sim[F:,C:])
                        
                        # Calculo la distancia en el segundo cuadrante (Derecha-abajo)
                        DKS[1] = np.sum(Distr_Enc[F:,:C])-np.sum(Distr_Sim[F:,:C])
                        
                        # Calculo la distancia en el tercer cuadrante (Izquierda-abajo)
                        DKS[2] = np.sum(Distr_Enc[:F,:C])-np.sum(Distr_Sim[:F,:C])
                        
                        # Calculo la distancia en el tercer cuadrante (Izquierda-arriba)
                        DKS[3] = np.sum(Distr_Enc[:F,C:])-np.sum(Distr_Sim[:F,C:])
                        
                        Mat_Dist[rotacion,F,C] = np.max(np.abs(DKS))
            
            # Una vez que calcule las 4 distancias habiendo rotado 4 veces la distribución,
            # lo que me queda es guardar eso en las matrices ZZ correspondientes.
            
            repeticion = int(Df_archivos.loc[Df_archivos["nombre"]==nombre,"iteracion"])
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = np.min(np.max(Mat_Dist,axis = (1,2)))
    
    ZZ_alterado = np.reshape(ZZ, (ZZ.shape[0]*ZZ.shape[1],ZZ.shape[2]))
    np.savetxt("../Matrices DKS/{}_vs_{}.csv".format(code_y,code_x), ZZ_alterado,delimiter = ",", fmt = "%.6f")

"""

#####################################################################################
#####################################################################################

"""
# Voy a construir una matriz que me permita ver la similaridad entre los clusters de
# preguntas definidos según distancia JS.

# Levanto los datos de los archivos de matrices. Puedo tomarlo de cualquier métrica
CarpMat=[[root,files] for root,dirs,files in os.walk("../Matrices DJS")]
Arc_Matrices = CarpMat[0][1]

# Armo mi dataframe en el cuál voy a guardar los datos de las similaridades
df_simil = pd.DataFrame(columns = Arc_Matrices, index = Arc_Matrices)

# Primero levanto los datos de la ANES
Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)

for arc_1 in Arc_Matrices[0:20]:
    
    # Defino los códigos x e y
    code_y = arc_1.strip(".csv").split("_")[0]
    code_x = arc_1.strip(".csv").split("_")[2]
    
    # Defino el peso asociado
    if code_x[3] == '1' and code_y[3] == '1':
        weights = 'V200010a'
    else:
        weights = 'V200010b'
    
    # Armo la tupla de la primer distribución
    tupla_1 = (code_x,code_y,weights)
    
    Enc_1 = func.Distrib_Anes(tupla_1, Df_ANES)
    Enc_1 = Enc_1.flatten()
    
    for arc_2 in Arc_Matrices[0:20]:
        
        # Defino los códigos x e y
        code_y = arc_2.strip(".csv").split("_")[0]
        code_x = arc_2.strip(".csv").split("_")[2]
        
        # Defino el peso asociado
        if code_x[3] == '1' and code_y[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
        
        # Armo la tupla de la primer distribución
        tupla_2 = (code_x,code_y,weights)
        
        # Levanto las dos distribuciones de los dos pares de preguntas
        Enc_2 = func.Distrib_Anes(tupla_2, Df_ANES)
        Enc_2 = Enc_2.flatten()
        
        # # Ahora que tengo las dos distribuciones, calculo las distancias de JS
        dist_previa = np.zeros(4)
        
        for rotacion in range(4):
            
            dist_previa[rotacion] = jensenshannon(Enc_1,Enc_2)
            
            # Una vez que hice el cálculo de la distancia y todo, roto la matriz
            Enc_2 = np.reshape(Enc_2, (6,6))
            Enc_2 = func.Rotar_matriz(Enc_2)
            Enc_2 = Enc_2.flatten()
        
        df_simil.loc[arc_1,arc_2] = np.min(dist_previa)

df_simil.to_csv("Dist_Enc_JS.csv")


#-------------------------------------------------------------------------------------------------------------------------------------

# Voy a construir una matriz que me permita ver la similaridad entre los clusters de
# preguntas definidos según distancia JS.

# Levanto los datos de los archivos de matrices. Puedo tomarlo de cualquier métrica
CarpMat=[[root,files] for root,dirs,files in os.walk("../Matrices DKS")]
Arc_Matrices = CarpMat[0][1]

# Armo mi dataframe en el cuál voy a guardar los datos de las similaridades
df_simil = pd.DataFrame(columns = Arc_Matrices, index = Arc_Matrices)

# Primero levanto los datos de la ANES
Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)

for arc_1 in Arc_Matrices:
    
    # Defino los códigos x e y
    code_y = arc_1.strip(".csv").split("_")[0]
    code_x = arc_1.strip(".csv").split("_")[2]
    
    # Defino el peso asociado
    if code_x[3] == '1' and code_y[3] == '1':
        weights = 'V200010a'
    else:
        weights = 'V200010b'
    
    # Armo la tupla de la primer distribución
    tupla_1 = (code_x,code_y,weights)
    
    Enc_1 = func.Distrib_Anes(tupla_1, Df_ANES)
    
    for arc_2 in Arc_Matrices:
        
        # Defino los códigos x e y
        code_y = arc_2.strip(".csv").split("_")[0]
        code_x = arc_2.strip(".csv").split("_")[2]
        
        # Defino el peso asociado
        if code_x[3] == '1' and code_y[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
        
        # Armo la tupla de la primer distribución
        tupla_2 = (code_x,code_y,weights)
        
        # Levanto las dos distribuciones de los dos pares de preguntas
        Enc_2 = func.Distrib_Anes(tupla_2, Df_ANES)
        
        Mat_Dist = np.zeros((4,6,6))
        for rotacion in range(4):
            
            for F in range(6):
                for C in range(6):
                    
                    DKS = np.zeros(4)
                    
                    # Calculo la distancia en el primer cuadrante (Derecha-arriba)
                    DKS[0] = np.sum(Enc_1[F:,C:])-np.sum(Enc_2[F:,C:])
                    
                    # Calculo la distancia en el segundo cuadrante (Derecha-abajo)
                    DKS[1] = np.sum(Enc_1[F:,:C])-np.sum(Enc_2[F:,:C])
                    
                    # Calculo la distancia en el tercer cuadrante (Izquierda-abajo)
                    DKS[2] = np.sum(Enc_1[:F,:C])-np.sum(Enc_2[:F,:C])
                    
                    # Calculo la distancia en el tercer cuadrante (Izquierda-arriba)
                    DKS[3] = np.sum(Enc_1[:F,C:])-np.sum(Enc_2[:F,C:])
                    
                    Mat_Dist[rotacion,F,C] = np.max(np.abs(DKS))
                    
            Enc_2 = func.Rotar_matriz(Enc_2)
        
        df_simil.loc[arc_1,arc_2] = np.min(np.max(Mat_Dist,axis = (1,2)))

df_simil.to_csv("Dist_Enc_KS.csv")

"""
#####################################################################################
#####################################################################################

"""

# Defino las preguntas del cluster de JS
Df_preguntas = pd.read_csv("Tabla_JS.csv")
Clusters = [(0,0.4), (0,0.6), (0.02,1.1), (0.08,1.1), (0.14,1.1), (0.48,0.4)]
preg_cluster = dict()

for tupla in Clusters:
    Cosd = tupla[0]
    Beta = tupla[1]
    
    preg_cluster[tupla] = Df_preguntas.loc[(Df_preguntas["Cosd_100"]==Cosd) & (Df_preguntas["Beta_100"]==Beta), "nombre"]

# Ya tengo la matriz de distancias de JS. A partir de esto puedo fácilmente reconstruir
# la distancia, así que es un poco lo mismo. Me gustaría entonces tomar este archivo y
# levantar cuáles son las distancias inter cluster e intra cluster. Primero para eso
# tengo que definir los clusters.

# Levanto los datos de la tabla de similaridad
Df_dist_JS = pd.read_csv("Dist_Enc_JS.csv", index_col=0)

#-------------------------------------------------------------------------------------------------------------------------------------


# Voy a armar gráficos de clusterización usando K-means
# para diversos números de clusters

# sse = [] # acá vamos a guardar el puntaje de la función objetivo
# silhouette_coefficients = [] # Acá guardo el puntaje de los coeficientes silhouette

for k in range(6,8):
    
    # Armo mi clusterizador y lo entreno
    kmeans = KMeans(n_clusters=k, random_state=42, n_init = "auto")
    kmeans.fit(X_pca)
    
    # Guardo las posiciones de los centroids
    centroids = kmeans.cluster_centers_
    
    plt.rcParams.update({'font.size': 44})
    plt.figure(figsize=(28, 21))  # Adjust width and height as needed
    scatter = plt.scatter(X_pca[:,0],X_pca[:,1], s=400, c = kmeans.labels_)
    # scatter = plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=800, linewidths=1,
    #             c=np.unique(kmeans.labels_), edgecolors='black')
    
    # Custom legend with specific text for each cluster
    legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(kmeans.labels_)]  # Customize these as you like
    # Create legend manually using custom text and colors from the scatter plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                          markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                          for i, label in enumerate(legend_labels)]
    # Add the legend to the plot
    plt.legend(handles=handles, loc="best", ncol=2)
    plt.title('Clusterización de K-means sobre PCA, {} Clusters, metrica JS'.format(k))
    direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/K-means_PCA_JS_k={}.png".format(k))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()
    
    # SSE suma de los cuadrados de la distancia euclidea de cada cluster
    # sse.append(kmeans.inertia_)
    
    # El silhouette score es un número que va entre -1 y 1. Si los clusters están
    # superpuestos, da -1. Si los clusters no se tocan, da 1.
    # silhouette_coefficients.append(silhouette_score(Df_dist_JS.to_numpy(), kmeans.labels_))


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
# estas lineas son el grafico de SSEvsK
plt.scatter(range(3, 11), sse, s=300)
plt.xlabel("Número de clusters")
plt.ylabel("SSE")
plt.title("Métrica JS")
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SSEk_PCA_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
# estas lineas son el grafico de SSEvsK
plt.scatter(range(3, 11), silhouette_coefficients, s=300)
plt.xlabel("Número de clusters")
plt.ylabel("Promedio coeficientes de Silhouette")
plt.title("Métrica JS")
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/CoefSilhouette_PCA_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

"""

#####################################################################################
#####################################################################################
"""
# Veamos de hacer el análisis de K-means sobre la matriz de similaridad entre todas las
# preguntas.

# Levanto los datos de la tabla de similaridad
Df_dist_JS = pd.read_csv("Dist_Enc_JS.csv", index_col=0)

kmeans = KMeans(n_clusters=7, random_state=42, n_init = "auto")
kmeans.fit(Df_dist_JS)

# Construyo mi operador que realizar un PCA sobre mis datos
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Df_dist_JS.to_numpy())

# Si quiero ver cuales son los autovalores de cada componente
###autovalores = pca.explained_variance_

# Si quiero ver cuál es la fracción de la varianza explicada por las
# componentes consideradas
###Var_acumulada = np.cumsum(pca.explained_variance_ratio_)

# Si quiero reconstruir los datos a partir de los datos reducidos en dimensionalidad
###X = pca.inverse_transform(X_pca)

# Esto arma el gráfico de distribución de puntos en 2D, sin clusterizar
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_pca[:,0],X_pca[:,1], s=400, marker = "s", color = "tab:red", alpha = 0.6)
plt.title('Reducción de dimensionalidad usando PCA')
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Distribucion_PCA_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#-------------------------------------------------------------------------------------------------------------------------------------

# Voy a querer construir un gráfico similar pero que tenga el coloreado según los clusters 
# construidos por la agrupación en espacio de parámetros según JS.

# Lo inicializo en 6 para que los no revisados sean un cluster aparte
Df_preguntas["clusters"] = 6

for cluster,tupla in enumerate(Clusters):
    Cosd = tupla[0]
    Beta = tupla[1]
    
    Df_preguntas.loc[(Df_preguntas["Cosd_100"]==Cosd) & (Df_preguntas["Beta_100"]==Beta), "clusters"] = cluster


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_pca[:,0],X_pca[:,1], s=400, c = Df_preguntas["clusters"])
plt.title('Clasificación en espacio de parámetros')
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(Df_preguntas["clusters"])]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_esp_parametros.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#####################################################################################
#####################################################################################

# Lo que se me ocurre es que puedo primero tener el dato de los clusters en un solo
# data frame.
# Guardo los datos según K-means en la matriz de distancia JS entre encuestas.

kmeans = KMeans(n_clusters=7, random_state=42, n_init = "auto")
kmeans.fit(Df_dist_JS)
Df_preguntas["pre_clusters_JS"] = kmeans.labels_

# Guardo los datos según K-means en aplicado al PCA de la matriz de distancia JS entre encuestas.

pca = PCA(n_components=2)
X_pca = pca.fit_transform(Df_dist_JS.to_numpy())
kmeans = KMeans(n_clusters=7, random_state=42, n_init = "auto")
kmeans.fit(X_pca)
Df_preguntas["pre_clusters_PCA_JS"] = kmeans.labels_

#-------------------------------------------------------------------------------------------------------------------------------------

# Lo siguiente es ir viendo cuáles clusters se parecen más a otros, y con eso
# ir luego viendo que estén todos pintados en consecuencia. Ese pintarse en
# consecuencia se resuelve con columnas nuevas

# Primero necesito ver de comparar conjuntos
Mat_sup = np.zeros((7,7))
for ic1 in range(np.unique(Df_preguntas["clusters"]).shape[0]):
    
    cluster_1 = Df_preguntas.loc[Df_preguntas["clusters"] == ic1, "nombre"]
    
    for ic2 in range(np.unique(Df_preguntas["clusters"]).shape[0]):
        
        cluster_2 = Df_preguntas.loc[Df_preguntas["pre_clusters_JS"] == ic2, "nombre"]
        
        Mat_sup[ic1, ic2] = (len(set(cluster_1) & set(cluster_2))) / (len(set(cluster_1) | set(cluster_2)))

# Lo siguiente es ir ordenando la matriz de forma correcta
Transicion = np.arange(Mat_sup.shape[0])
for fila in range(Mat_sup.shape[0]):
    
    i_cambio = np.argmax(Mat_sup[fila,fila:])
    Transicion[[fila, fila + i_cambio]] = Transicion[[fila + i_cambio, fila]]
    Mat_sup[:,[fila, fila + i_cambio]] = Mat_sup[:,[fila+i_cambio, fila]]

# Ahora que lo tengo ordenado, construyo la nueva clusterización
Df_preguntas["clusters_JS_ord"] = None
for ic,cluster in enumerate(Transicion):
    Df_preguntas.loc[Df_preguntas["pre_clusters_JS"]==ic, "clusters_JS_ord"] = cluster


# Create row and column labels
row_labels = ["Clust {} ({})".format(k, np.unique(Df_preguntas["clusters"],return_counts=True)[1][k-1]) for k in range(1,8)]
col_labels = ["Clust {} \n ({})".format(k, np.unique(Df_preguntas["clusters_JS_ord"],return_counts=True)[1][k-1]) for k in range(1,8)]
# Plot the colormap using imshow
plt.rcParams.update({'font.size': 38})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
im = plt.imshow(Mat_sup, cmap='RdYlBu', aspect='auto')
# Add colorbar
cbar = plt.colorbar(im)
im.set_clim(0,1)
# Add column text labels
plt.xticks(ticks=np.arange(Mat_sup.shape[1]), labels=col_labels)
# Add row text labels
plt.yticks(ticks=np.arange(Mat_sup.shape[0]), labels=row_labels)
# Display the plot
plt.title("Índice de Jaccard")
plt.ylabel("Clust. Datos")
plt.xlabel("Clust. Matriz")
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Sup_clusters_Ord.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


#-------------------------------------------------------------------------------------------------------------------------------------

# Repito esto para los clusters obtenidos de la proyección 2D por PCA

# Primero necesito ver de comparar conjuntos
Mat_sup = np.zeros((7,7))
for ic1 in range(np.unique(Df_preguntas["clusters"]).shape[0]):
    
    cluster_1 = Df_preguntas.loc[Df_preguntas["clusters"] == ic1, "nombre"]
    
    for ic2 in range(np.unique(Df_preguntas["clusters"]).shape[0]):
        
        cluster_2 = Df_preguntas.loc[Df_preguntas["pre_clusters_PCA_JS"] == ic2, "nombre"]
        
        Mat_sup[ic1, ic2] = (len(set(cluster_1) & set(cluster_2))) / (len(set(cluster_1) | set(cluster_2)))

# Lo siguiente es ir ordenando la matriz de forma correcta
Transicion = np.arange(Mat_sup.shape[0])
for fila in range(Mat_sup.shape[0]):
    
    i_cambio = np.argmax(Mat_sup[fila,fila:])
    Transicion[[fila, fila + i_cambio]] = Transicion[[fila + i_cambio, fila]]
    Mat_sup[:,[fila, fila + i_cambio]] = Mat_sup[:,[fila+i_cambio, fila]]

# Ahora que lo tengo ordenado, construyo la nueva clusterización
Df_preguntas["clusters_PCA_JS_ord"] = None
for ic,cluster in enumerate(Transicion):
    Df_preguntas.loc[Df_preguntas["pre_clusters_PCA_JS"]==ic, "clusters_PCA_JS_ord"] = cluster

# Create row and column labels
row_labels = ["Clust {} ({})".format(k, np.unique(Df_preguntas["clusters"],return_counts=True)[1][k-1]) for k in range(1,8)]
col_labels = ["Clust {} \n ({})".format(k, np.unique(Df_preguntas["clusters_PCA_JS_ord"],return_counts=True)[1][k-1]) for k in range(1,8)]
# Plot the colormap using imshow
plt.rcParams.update({'font.size': 38})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
im = plt.imshow(Mat_sup, cmap='RdYlBu', aspect='auto')
# Add colorbar
cbar = plt.colorbar(im)
im.set_clim(0,1)
# Add column text labels
plt.xticks(ticks=np.arange(Mat_sup.shape[1]), labels=col_labels)
# Add row text labels
plt.yticks(ticks=np.arange(Mat_sup.shape[0]), labels=row_labels)
# Display the plot
plt.title("Índice Jaccard")
plt.ylabel("Clust. Datos")
plt.xlabel("Clust. Proyección Matriz")
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Sup_clusters_PCA_Ord.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#-------------------------------------------------------------------------------------------------------------------------------------

# Hago una comparación entre las particiones obtenidas de los K-Means

# Primero necesito ver de comparar conjuntos
Mat_sup = np.zeros((7,7))
for ic1 in range(np.unique(Df_preguntas["clusters"]).shape[0]):
    
    cluster_1 = Df_preguntas.loc[Df_preguntas["clusters_JS_ord"] == ic1, "nombre"]
    
    for ic2 in range(np.unique(Df_preguntas["clusters"]).shape[0]):
        
        cluster_2 = Df_preguntas.loc[Df_preguntas["pre_clusters_PCA_JS"] == ic2, "nombre"]
        
        Mat_sup[ic1, ic2] = (len(set(cluster_1) & set(cluster_2))) / (len(set(cluster_1) | set(cluster_2)))

# Lo siguiente es ir ordenando la matriz de forma correcta
Transicion = np.arange(Mat_sup.shape[0])
for fila in range(Mat_sup.shape[0]):
    
    i_cambio = np.argmax(Mat_sup[fila,fila:])
    Transicion[[fila, fila + i_cambio]] = Transicion[[fila + i_cambio, fila]]
    Mat_sup[:,[fila, fila + i_cambio]] = Mat_sup[:,[fila+i_cambio, fila]]

# Ahora que lo tengo ordenado, construyo la nueva clusterización
Df_preguntas["clusters_PCA_JS_neword"] = None
for ic,cluster in enumerate(Transicion):
    Df_preguntas.loc[Df_preguntas["pre_clusters_PCA_JS"]==ic, "clusters_PCA_JS_neword"] = cluster

# Create row and column labels
row_labels = ["Clust {} ({})".format(k, np.unique(Df_preguntas["clusters_JS_ord"],return_counts=True)[1][k-1]) for k in range(1,8)]
col_labels = ["Clust {} \n ({})".format(Transicion[k-1]+1, np.unique(Df_preguntas["clusters_PCA_JS_ord"],return_counts=True)[1][Transicion[k-1]]) for k in range(1,8)]
# Plot the colormap using imshow
plt.rcParams.update({'font.size': 38})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
im = plt.imshow(Mat_sup, cmap='RdYlBu', aspect='auto')
# Add colorbar
cbar = plt.colorbar(im)
im.set_clim(0,1)
# Add column text labels
plt.xticks(ticks=np.arange(Mat_sup.shape[1]), labels=col_labels)
# Add row text labels
plt.yticks(ticks=np.arange(Mat_sup.shape[0]), labels=row_labels)
# Display the plot
plt.title("Índice Jaccard")
plt.xlabel("Clust. Proyección Matriz")
plt.ylabel("Clust. Matriz")
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Sup_clusters_kmeans_Ord.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()



#-------------------------------------------------------------------------------------------------------------------------------------

# Ahora los grafico a los dos

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_pca[:,0],X_pca[:,1], s=400, c = Df_preguntas["clusters_JS_ord"])
plt.title('Clusterización Matriz Distancia')
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(Df_preguntas["clusters_JS_ord"])]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_Mat_dist.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_pca[:,0],X_pca[:,1], s=400, c = Df_preguntas["clusters_PCA_JS_ord"])
plt.title('Clusterización aplicado al PCA')
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(Df_preguntas["clusters_PCA_JS_ord"])]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_dist_PCA.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#-------------------------------------------------------------------------------------------------------------------------------------

# Una vez que tengo armados mis clusterings, uso NMI y rand_index ajustado para
# comparar qué tan similares son esos clusterings.

# Voy a guardar la info en arrays de tamaño tres, en el primer elemento pongo 
# La clust. de los datos vs la clust de la matriz de JS, en el segundo pongo la
# clust de los datos versus la clust de la proyección de la matriz de JS y en la
# tercer componente pongo la clust de la matriz versus la de la proyección.

info_mutua = np.zeros(3)
info_mutua[0] = normalized_mutual_info_score(Df_preguntas["clusters"], Df_preguntas["clusters_JS_ord"])
info_mutua[1] = normalized_mutual_info_score(Df_preguntas["clusters"], Df_preguntas["clusters_PCA_JS_ord"])
info_mutua[2] = normalized_mutual_info_score(Df_preguntas["clusters_JS_ord"], Df_preguntas["clusters_PCA_JS_neword"])


ind_rand = np.zeros(3)
ind_rand[0] = adjusted_rand_score(Df_preguntas["clusters"], Df_preguntas["clusters_JS_ord"])
ind_rand[1] = adjusted_rand_score(Df_preguntas["clusters"], Df_preguntas["clusters_PCA_JS_ord"])
ind_rand[2] = adjusted_rand_score(Df_preguntas["clusters_JS_ord"], Df_preguntas["clusters_PCA_JS_neword"])


#####################################################################################
#####################################################################################
"""


# Voy a querer clusterizar sobre los distintos espacios de parámetros, y luego comparar
# esos clusters con los clusters obtenidos al clusterizar la Mat_Dist_JS.
# Para comparar esos clusters, en principio voy a usar simplemente rand_index o mutual info
# para compararlos, como para ser rápido en la idea.

# Levanto los datos de la tabla de similaridad
Df_dist_JS = pd.read_csv("Dist_Enc_JS.csv", index_col=0)
# Defino las preguntas del cluster de JS
Df_preguntas = pd.read_csv("Tabla_JS.csv")

# Hago el PCA de los datos
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Df_dist_JS.to_numpy())

# Armo un rng para agregar un ruido normal
rng = np.random.default_rng()

num_clusters = np.arange(3,11)
num_simul = np.arange(1,11)*10

XX,YY = np.meshgrid(num_clusters,num_simul)

# Hago la comparación de clusters usando Rand_index
ZZ = np.zeros(XX.shape)
# for columna,k in enumerate(num_clusters):
    
#     # Armo mi clusterizador y lo entreno para detectar clusters en matriz de 
#     # distancias entre gráficos
#     kmeans_mat = KMeans(n_clusters=k, random_state=42, n_init = "auto")
#     kmeans_mat.fit(Df_dist_JS.to_numpy()) # + rng.normal(loc = 0, scale = 0.004, size = Df_dist_JS.to_numpy().shape))
    
#     for fila,rank in enumerate(range(10)):
        
#         # Armo mi clusterizador y lo entreno para detectar clusters en espacio
#         # de parámetros
#         kmeans_esp = KMeans(n_clusters=k, random_state=42, n_init = "auto")
#         kmeans_esp.fit(Df_preguntas[["Cosd_{}".format(rank*10+10),"Beta_{}".format(rank*10+10)]])
        
#         ZZ[fila,columna] = adjusted_rand_score(kmeans_esp.labels_, kmeans_mat.labels_)

# # Armo los parámetros del gráfico de promedios
# plt.rcParams.update({'font.size': 44})
# plt.figure("Rindex clusters",figsize=(28,21))
# plt.xlabel("Cantidad de clusters")
# plt.ylabel("Cantidad de simulaciones")
# plt.title("Rand Index ajustado de clusterizaciones")

# # Guardo la figura
# direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Randindex_clusters.png")
# plt.pcolormesh(XX,YY,ZZ, shading="nearest", cmap = "viridis")
# plt.colorbar()
# # Guardo la figura y la cierro
# plt.savefig(direccion_guardado , bbox_inches = "tight")
# plt.close("Rindex clusters")

# hago la comparación de clusters usando mutual informaiton
ZZ = np.zeros(XX.shape)
for columna,k in enumerate(num_clusters):
    
    # Armo mi clusterizador y lo entreno para detectar clusters en matriz de 
    # distancias entre gráficos
    kmeans_mat = KMeans(n_clusters=k, random_state=42, n_init = "auto")
    kmeans_mat.fit(Df_dist_JS.to_numpy()) # + rng.normal(loc = 0, scale = 0.004, size = Df_dist_JS.to_numpy().shape))
    
    for fila,rank in enumerate(range(10)):
        
        # Armo mi clusterizador y lo entreno para detectar clusters en espacio
        # de parámetros
        kmeans_esp = KMeans(n_clusters=k, random_state=42, n_init = "auto")
        kmeans_esp.fit(Df_preguntas[["Cosd_{}".format(rank*10+10),"Beta_{}".format(rank*10+10)]])
        
        ZZ[fila,columna] = normalized_mutual_info_score(kmeans_esp.labels_, kmeans_mat.labels_)

# Armo los parámetros del gráfico de promedios
plt.rcParams.update({'font.size': 44})
plt.figure("Minfo clusters",figsize=(28,21))
plt.xlabel("Cantidad de clusters")
plt.ylabel("Cantidad de simulaciones")
plt.title("Mutual info. normalizada de clusterizaciones")

# Guardo la figura
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Minfo_clusters.png")
plt.pcolormesh(XX,YY,ZZ, shading="nearest", cmap = "viridis")
plt.colorbar()
# Guardo la figura y la cierro
plt.savefig(direccion_guardado , bbox_inches = "tight")
plt.close("Minfo clusters")
"""
#---------------------------------------------------------------------------------------------

# Habiendo encontrado las clusterizaciones que maximizan similaridad, grafiquémoslas

# Primero armo la clusterización adecuada
kmeans_esp = KMeans(n_clusters=5, random_state=42, n_init = "auto")
kmeans_esp.fit(Df_preguntas[["Cosd_{}".format(90),"Beta_{}".format(90)]])

# Grafico la clusterización del espacio de parámetros en el espacio proyectado de PCA
# plt.rcParams.update({'font.size': 44})
# plt.figure(figsize=(28, 21))  # Adjust width and height as needed
# scatter = plt.scatter(X_pca[:,0],X_pca[:,1], s=400, c = kmeans_esp.labels_)
# plt.title('Kmeans en espacio de parámetros, 90 simul más similares')
# # Custom legend with specific text for each cluster
# legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(kmeans_esp.labels_)]  # Customize these as you like
# # Create legend manually using custom text and colors from the scatter plot
# handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
#                       markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
#                       for i, label in enumerate(legend_labels)]
# # Add the legend to the plot
# plt.legend(handles=handles, loc="best", ncol=2)
# direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clust_max_simil_esp_parametros.png")
# plt.savefig(direccion_guardado ,bbox_inches = "tight")
# plt.close()

# Lo que queda es hacer los plot scatter de las preguntas
X = Df_preguntas["Cosd_{}".format(90)] + rng.normal(loc = 0, scale = (0.02)/5, size = Df_preguntas["Cosd_{}".format(90)].shape)
Y = Df_preguntas["Beta_{}".format(90)] + rng.normal(loc = 0, scale = (0.1)/5, size = Df_preguntas["Beta_{}".format(90)].shape)

plt.figure(figsize=(28, 21))  # Adjust width and height as needed
tlinea = 5
# Región de Polarización Descorrelacionada
x = [0, 0.1, 0.15, 0, 0]  # x-coordinates
y = [1.1, 1.1, 1.5, 1.5, 1.1]  # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4) #, label='Polarización Descorrelacionada')
plt.text(0.05, 1.3, 'I', fontsize=40, ha='center', va='center', color='k')
# Región de Transición
x = [0.1, 0.15, 0.3, 0.15, 0.1]  # x-coordinates
y = [1.1, 1.1, 1.5, 1.5, 1.1]  # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4)
plt.text(0.18, 1.3, 'II', fontsize=40, ha='center', va='center', color='k')
# Región de Polarización ideológica
x = [0.15, 0.5, 0.5, 0.3, 0.15] # x-coordinates
y = [1.1, 1.1, 1.5, 1.5, 1.1] # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4)
plt.text(0.35, 1.3, 'III', fontsize=40, ha='center', va='center', color='k')
# Región de Consenso Radicalizado
x = [0, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.1, 0.1, 0, 0]  # x-coordinates
y = [0, 0, 0.15, 0.3, 0.6, 0.6, 1.1, 1.1, 0.3, 0.2, 0]  # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4)
plt.text(0.3, 0.85, 'VI', fontsize=40, ha='center', va='center', color='k')
# Región de Mezcla 1
x = [0, 0.1, 0.1, 0, 0] # x-coordinates
y = [0.2, 0.3, 0.75, 0.75, 0.2] # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4)  # label=r'Mezcla: CR (50~80%), P1Da (20~35%)')
plt.text(0.05, 0.55, 'V', fontsize=40, ha='center', va='center', color='k')
# Región de Mezcla 2
x = [0, 0.1, 0.1, 0, 0] # x-coordinates
y = [0.75, 0.75, 1.1, 1.1, 0.75] # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4) # label=r'Mezcla: CR (30~40%), P1D (10~50%)')
plt.text(0.05, 0.9, 'IV', fontsize=40, ha='center', va='center', color='k')
# Región de Mezcla 3
x = [0.2, 0.5, 0.5, 0.2, 0.2] # x-coordinates
y = [0.3, 0.15, 0.6, 0.6, 0.3] # y-coordinates
plt.plot(x, y, color='k', linestyle = "dashed", linewidth=tlinea, alpha = 0.4) # label=r'Mezcla: CR (40~80%) y PIa (10~45%)')
plt.text(0.3, 0.45, 'VII', fontsize=40, ha='center', va='center', color='k')
plt.scatter(X,Y, marker="o", c = kmeans_esp.labels_ , s = 500,cmap = "tab10" , alpha = 0.7)
plt.xlabel(r"$cos(\delta)$")
plt.ylabel(r"$\beta$")
plt.xlim(-0.025,0.525)
plt.ylim(0,1.55)
plt.title("90 simulaciones, Dist {}".format("JS"))
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Esp_parametros_Max_Simil_Clust.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()



# Primero armo la clusterización adecuada
# kmeans_esp = KMeans(n_clusters=4, random_state=42, n_init = "auto")
# kmeans_esp.fit(Df_dist_JS.to_numpy())
# # Grafico la clusterización del espacio de parámetros en el espacio proyectado de PCA
# plt.rcParams.update({'font.size': 44})
# plt.figure(figsize=(28, 21))  # Adjust width and height as needed
# scatter = plt.scatter(X_pca[:,0],X_pca[:,1], s=400, c = kmeans_esp.labels_)
# plt.title('Kmeans en Matriz Dist JS')
# # Custom legend with specific text for each cluster
# legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(kmeans_esp.labels_)]  # Customize these as you like
# # Create legend manually using custom text and colors from the scatter plot
# handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
#                       markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
#                       for i, label in enumerate(legend_labels)]
# # Add the legend to the plot
# plt.legend(handles=handles, loc="best", ncol=2)
# direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clust_max_simil_JS.png")
# plt.savefig(direccion_guardado ,bbox_inches = "tight")
# plt.close()

"""

#####################################################################################
#####################################################################################

# Primero voy a intentar encontrar los verdaderos clusters que se hallan en los datos
# del espacio de Matriz de Dist JS. Para eso tengo que probar diversas formas de
# clusterizar y métricas para analizarlas

# Levanto los datos de la tabla de similaridad
Df_dist_JS = pd.read_csv("Dist_Enc_JS.csv", index_col=0)
# Defino las preguntas del cluster de JS
Df_preguntas = pd.read_csv("Tabla_JS.csv")

# Armo un rng para agregar un ruido normal
rng = np.random.default_rng()

# Reduzco dimensionalidad usando Multi-Dimensional Scaling
mds = MDS(n_components = 2, random_state = 42, dissimilarity = "precomputed", normalized_stress = "auto")
X_2d = mds.fit_transform(Df_dist_JS.to_numpy())

# Grafico los datos en el espacio reducido usando MDS
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_2d[:,0],X_2d[:,1], s=400, marker = "8", color = "tab:orange")
plt.title('MDS Matriz JS')
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

"""
# Reduzco dimensionalidad usando Principal Component Analysis
# Construyo mi operador que realizar un PCA sobre mis datos
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Df_dist_JS.to_numpy())

# Esto arma el gráfico de distribución de puntos en 2D, sin clusterizar
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_pca[:,0],X_pca[:,1], s=400, marker = "s", color = "tab:red")
plt.title('PCA matriz JS')
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/PCA_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()
"""
#---------------------------------------------------------------------------------------------

# Ahora que tengo los gráficos ubicados en el espacio 2D, veamos de armar clusterizaciones
# y usar métricas para ver cuál es la cantidad correcta de clusters a observar. Una vez
# hecho eso, graficaré la "mejor" clusterización
"""
# Primero clasifico usando DBSCAN
radios = np.linspace(0.4,0.6,100)
silhouette_scores = list()
dabo_scores = list()

for dist in radios:
    # Clusterizo
    dbscan = DBSCAN(eps=dist, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
    dbscan.fit(Df_dist_JS.to_numpy())
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(Df_dist_JS.to_numpy(), dbscan.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # Calculate the Davies_Bouldin score
    dabo_scores.append(davies_bouldin_score(Df_dist_JS.to_numpy(), dbscan.labels_))

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN Mat JS')
plt.ylabel("Silhouette Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_DBSCAN_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,dabo_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN Mat JS')
plt.ylabel("Davies-Bouldin Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/DaboS_DBSCAN_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()
"""

#---------------------------------------------------------------------------------------------

# Segundo clasifico usando K-means
cant_clusters = np.arange(2,13)
silhouette_scores = list()
sse = list() # acá vamos a guardar el puntaje de la función objetivo

for clusters in cant_clusters:
    # Clusterizo
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init = "auto")
    kmeans.fit(Df_dist_JS.to_numpy())
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(Df_dist_JS.to_numpy(), kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # SSE suma de los cuadrados de la distancia euclidea de cada cluster
    sse.append(kmeans.inertia_)

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(cant_clusters,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización K-means sobre Mat JS')
plt.ylabel("Silhouette Score")
plt.xlabel("Cant. Clusters")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_Kmeans_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(cant_clusters,sse, linewidth=6, color = "tab:blue")
plt.title('Caracterización K-means sobre Mat JS')
plt.ylabel("SSE")
plt.xlabel("Cant. Clusters")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SSE_Kmeans_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

kmeans = KMeans(n_clusters=5, random_state=42, n_init = "auto")
kmeans.fit(Df_dist_JS.to_numpy())

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = kmeans.labels_, cmap = "tab10")
plt.title('Silh {}, K-means sobre Mat JS'.format(round(silhouette_scores[3],2)))
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(kmeans.labels_)]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_silh_K-means5_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#---------------------------------------------------------------------------------------------

# Hago una prueba de clusterizar los datos en el espacio 2D y ver si el score
# de Silhouette tiene sentido. Clusterizo sobre el espacio 2D
"""
# Primero clasifico usando DBSCAN
radios = np.linspace(0.03,0.07,100)
silhouette_scores = list()
dabo_scores = list()

for dist in radios:
    # Clusterizo
    dbscan = DBSCAN(eps=dist, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
    dbscan.fit(X_2d)
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X_2d, dbscan.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # Calculate the Davies_Bouldin score
    dabo_scores.append(davies_bouldin_score(X_2d, dbscan.labels_))


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN sobre MDS de Mat JS')
plt.ylabel("Silhouette Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_DBSCAN_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,dabo_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN sobre MDS de Mat JS')
plt.ylabel("Davies-Bouldin Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/DaboS_DBSCAN_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


dbscan = DBSCAN(eps=radios[np.argmax(silhouette_scores)], min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
dbscan.fit(X_2d)
labels = dbscan.labels_
labels[labels == -1] = np.max(labels)+1

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = labels, cmap = "tab10")
plt.title('Max. Silh, eps={} DBSCAN sobre MDS de Mat JS'.format(round(radios[np.argmax(silhouette_scores)],4)))
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(labels)]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_maxsilh_DBSCAN_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()
"""

"""
dbscan = DBSCAN(eps=radios[np.argmin(dabo_scores)], min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
dbscan.fit(X_2d)
labels = dbscan.labels_
labels[labels == -1] = np.max(labels)+1

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = labels, cmap = "tab10")
plt.title('Min. Dabo, eps={} DBSCAN sobre MDS de Mat JS'.format(round(radios[np.argmin(dabo_scores)],4)))
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(labels)]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_mindabo_DBSCAN_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()
"""

#---------------------------------------------------------------------------------------------
"""
# Segundo clasifico usando K-means
cant_clusters = np.arange(2,13)
silhouette_scores = list()
sse = list() # acá vamos a guardar el puntaje de la función objetivo

for clusters in cant_clusters:
    # Clusterizo
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init = "auto")
    kmeans.fit(X_2d)
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X_2d, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # SSE suma de los cuadrados de la distancia euclidea de cada cluster
    sse.append(kmeans.inertia_)


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(cant_clusters,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización K-means sobre MDS de Mat JS')
plt.ylabel("Silhouette Score")
plt.xlabel("Cant. Clusters")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_K-means_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(cant_clusters,sse, linewidth=6, color = "tab:blue")
plt.title('Caracterización K-means sobre MDS de Mat JS')
plt.ylabel("SSE")
plt.xlabel("Cant. Clusters")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SSE_K-means_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


kmeans = KMeans(n_clusters=cant_clusters[np.argmax(silhouette_scores)], random_state=42, n_init = "auto")
kmeans.fit(X_2d)

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = kmeans.labels_, cmap = "tab10")
plt.title('Max. Silh {}, K-means sobre MDS de Mat JS'.format(round(np.max(silhouette_scores),4)))
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(kmeans.labels_)]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_maxsilh_K-means_MDS_JS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


# dbscan = DBSCAN(eps=radios[np.argmax(silhouette_scores)], min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
# dbscan.fit(X_2d)
# labels = dbscan.labels_
# labels[labels == -1] = np.max(labels)+1

# plt.rcParams.update({'font.size': 44})
# plt.figure(figsize=(28, 21))  # Adjust width and height as needed
# scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = labels, cmap = "tab10")
# plt.title('Max. Silh, eps={} DBSCAN sobre MDS de Mat JS'.format(round(radios[np.argmax(silhouette_scores)],4)))
# # Custom legend with specific text for each cluster
# legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(labels)]  # Customize these as you like
# # Create legend manually using custom text and colors from the scatter plot
# handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
#                       markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
#                       for i, label in enumerate(legend_labels)]
# # Add the legend to the plot
# plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
# direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_maxsilh_DBSCAN_MDS_JS.png")
# plt.savefig(direccion_guardado ,bbox_inches = "tight")
# plt.close()
"""
#---------------------------------------------------------------------------------------------

# Ahora comparo estas dos particiones

# info_mutua = normalized_mutual_info_score(dbscan.labels_, kmeans.labels_)
# print("La info mutua entre ambas particiones es: ", info_mutua)

# ind_rand = adjusted_rand_score(dbscan.labels_, kmeans.labels_)
# print("El rand_index entre ambas particiones es: ", ind_rand)

#---------------------------------------------------------------------------------------------

# Por último, comparo ambas clusterizaciones con clusterizaciones en el espacio de parámetros

X = np.arange(1,11)*10

# Armo las clusterizaciones correctas según los gráficos de Coef. de Silhouette
kmeans = KMeans(n_clusters=5, random_state=42, n_init = "auto")
kmeans.fit(Df_dist_JS.to_numpy())

# dbscan = DBSCAN(eps=0.525, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
# dbscan.fit(Df_dist_JS.to_numpy())


# Armo los valores de Y a graficar.
Y_KM = np.zeros(X.shape)
# Y_DB = np.zeros(X.shape)
for i,rank in enumerate(X):
    
    # Armo mi clusterizador y lo entreno para detectar clusters en espacio
    # de parámetros
    kmeans_esp = KMeans(n_clusters=5, random_state=42, n_init = "auto")
    kmeans_esp.fit(Df_preguntas[["Cosd_{}".format(rank),"Beta_{}".format(rank)]])
    
    # dbscan_esp = DBSCAN(eps=0.525, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
    # dbscan_esp.fit(Df_preguntas[["Cosd_{}".format(rank),"Beta_{}".format(rank)]])
    
    Y_KM[i] = normalized_mutual_info_score(kmeans.labels_, kmeans_esp.labels_)
    # Y_DB[i] = normalized_mutual_info_score(dbscan.labels_, dbscan_esp.labels_)


# Grafico las curvas de similaridad entre las particiones
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(X,Y_KM, "k--" ,linewidth = 6, label = "K-means, 5 clusters")
# plt.plot(X,Y_DB, "g--" ,linewidth = 6, label = "DBSCAN, eps = 0.525")
plt.title('Clust. Matriz JS vs Clust. espacio parámetros, MI')
plt.xlabel("Cant. Simulaciones")
plt.ylabel("Normalized Mutual Info")
plt.legend()
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/NI_comparacion_clusterizaciones.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

"""
# Armo los valores de Y a graficar.
Y_KM = np.zeros(X.shape)
# Y_DB = np.zeros(X.shape)
for i,rank in enumerate(X):
    
    # Armo mi clusterizador y lo entreno para detectar clusters en espacio
    # de parámetros
    kmeans_esp = KMeans(n_clusters=5, random_state=42, n_init = "auto")
    kmeans_esp.fit(Df_preguntas[["Cosd_{}".format(rank),"Beta_{}".format(rank)]])
    
    dbscan_esp = DBSCAN(eps=0.525, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
    dbscan_esp.fit(Df_preguntas[["Cosd_{}".format(rank),"Beta_{}".format(rank)]])
    
    Y_KM[i] = adjusted_rand_score(kmeans.labels_, kmeans_esp.labels_)
    # Y_DB[i] = adjusted_rand_score(dbscan.labels_, dbscan_esp.labels_)


# Grafico las curvas de similaridad entre las particiones
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(X,Y_KM, "r--" ,linewidth = 6, label = "K-means, 5 clusters")
# plt.plot(X,Y_DB, "g--" ,linewidth = 6, label = "DBSCAN, eps = 0.525")
plt.title('Clust. Matriz JS vs Clust. espacio parámetros, ARI')
plt.xlabel("Cant. Simulaciones")
plt.ylabel("Adjusted Rand Index")
plt.legend()
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/ARI_comparacion_clusterizaciones.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()
"""
#####################################################################################
#####################################################################################

"""
# Voy a intentar ordenar todos los histogramas para que se roten de forma de que
# queden alineados. Voy a intentar alinearlos usando cuadrados mínimos.

# Primero levanto los datos de la ANES
Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)

# Construyo la distribución de mi gráfico de referencia
code_1 = "V201408x"
code_2 = "V201372x"
weights = "V200010a"

Enc_ref = func.Distrib_Anes((code_1,code_2,weights), Df_ANES)

#---------------------------------------------------------------------------------------------

# Ya levanté el archivo que va a ser la referencia. Ahora necesito levantar
# el resto y construir la matriz de 120x36

# Defino las preguntas del cluster de JS
Df_preguntas = pd.read_csv("Tabla_JS.csv")
Mat_final = np.zeros((120,36))

for fila,code_1,code_2 in zip(np.arange(120),Df_preguntas["código y"],Df_preguntas["código x"]):
    
    # Defino el peso asociado
    if code_1[3] == '1' and code_2[3] == '1':
        weights = 'V200010a'
    else:
        weights = 'V200010b'
    
    Enc_obs = func.Distrib_Anes((code_1,code_2,weights), Df_ANES)
    cuad_min = np.zeros(4)
    
    for rot in range(4):
        cuad_min[rot] = np.sum((Enc_ref.flatten() - Enc_obs.flatten())**2)
        Enc_obs = func.Rotar_matriz(Enc_obs)
    
    rotaciones = np.argmin(cuad_min)
    for rot in range(rotaciones):
        Enc_obs = func.Rotar_matriz(Enc_obs)
    
    Mat_final[fila] = Enc_obs.flatten()

df_rotado = pd.DataFrame(Mat_final)
df_rotado.insert(0,"nombre",Df_preguntas["nombre"])
df_rotado.to_csv("Distribuciones_alineadas.csv", index = False)
"""

#####################################################################################
#####################################################################################

"""

# Levanto la matriz de las distribuciones de las encuestas alineadas
Df_enc_alin = pd.read_csv("Distribuciones_alineadas.csv", index_col=0)
# Ahora hago todo el bardo de clusterización en este espacio

# Defino las preguntas del cluster de JS
Df_preguntas = pd.read_csv("Tabla_JS.csv")

# Reduzco dimensionalidad usando Multi-Dimensional Scaling
mds = MDS(n_components = 2, random_state = 42, normalized_stress = "auto")
X_2d = mds.fit_transform(Df_enc_alin.to_numpy())

# Grafico los datos en el espacio reducido usando MDS
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_2d[:,0],X_2d[:,1], s=400, marker = "8", color = "tab:orange")
plt.title('MDS esp. de distribuciones')
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Distribucion_MDS.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


# Reduzco dimensionalidad usando Principal Component Analysis
# Construyo mi operador que realizar un PCA sobre mis datos
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Df_enc_alin.to_numpy())

# Esto arma el gráfico de distribución de puntos en 2D, sin clusterizar
plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.scatter(X_pca[:,0],X_pca[:,1], s=400, marker = "s", color = "tab:red")
plt.title('PCA esp. de distribuciones')
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Distribucion_PCA.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#---------------------------------------------------------------------------------------------

# Ahora que tengo los gráficos ubicados en el espacio 2D, veamos de armar clusterizaciones
# y usar métricas para ver cuál es la cantidad correcta de clusters a observar. Una vez
# hecho eso, graficaré la "mejor" clusterización

# Primero clasifico usando DBSCAN
radios = np.linspace(0.05,0.1,100)
silhouette_scores = list()
dabo_scores = list()

for dist in radios:
    # Clusterizo
    dbscan = DBSCAN(eps=dist, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
    dbscan.fit(Df_enc_alin.to_numpy())
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(Df_enc_alin.to_numpy(), dbscan.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # Calculate the Davies_Bouldin score
    dabo_scores.append(davies_bouldin_score(Df_enc_alin.to_numpy(), dbscan.labels_))


plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN sobre Mat distrib.')
plt.ylabel("Silhouette Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_DBSCAN_Distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,dabo_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN sobre Mat distrib.')
plt.ylabel("Davies-Bouldin Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/DaboS_DBSCAN_Distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#---------------------------------------------------------------------------------------------

# Segundo clasifico usando K-means
cant_clusters = np.arange(2,16)
silhouette_scores = list()
sse = list() # acá vamos a guardar el puntaje de la función objetivo

for clusters in cant_clusters:
    # Clusterizo
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init = "auto")
    kmeans.fit(Df_enc_alin.to_numpy())
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(Df_enc_alin.to_numpy(), kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # SSE suma de los cuadrados de la distancia euclidea de cada cluster
    sse.append(kmeans.inertia_)

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(cant_clusters,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización K-means sobre Mat distrib.')
plt.ylabel("Silhouette Score")
plt.xlabel("Cant. Clusters")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_Kmeans_Distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(cant_clusters,sse, linewidth=6, color = "tab:blue")
plt.title('Caracterización K-means sobre Mat distrib.')
plt.ylabel("SSE")
plt.xlabel("Cant. Clusters")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SSE_Kmeans_Distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

#---------------------------------------------------------------------------------------------

# Hago una prueba de clusterizar los datos en el espacio 2D y ver si el score
# de Silhouette tiene sentido. Clusterizo sobre el espacio 2D

# Primero clasifico usando DBSCAN
radios = np.linspace(0.03,0.07,100)
silhouette_scores = list()
dabo_scores = list()

for dist in radios:
    # Clusterizo
    dbscan = DBSCAN(eps=dist, min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
    dbscan.fit(X_2d)
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X_2d, dbscan.labels_)
    silhouette_scores.append(silhouette_avg)
    
    # Calculate the Davies_Bouldin score
    dabo_scores.append(davies_bouldin_score(Df_enc_alin.to_numpy(), dbscan.labels_))

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,silhouette_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN MDS de Mat distrib.')
plt.ylabel("Silhouette Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/SilhouetteS_DBSCAN_MDS_Distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
plt.plot(radios,dabo_scores, linewidth=6, color = "tab:blue")
plt.title('Caracterización DBSCAN MDS de Mat distrib.')
plt.ylabel("Davies-Bouldin Score")
plt.xlabel("Distancia máxima entre muestras")
plt.grid()
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/DaboS_DBSCAN_MDS_Distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


dbscan = DBSCAN(eps=radios[np.argmax(silhouette_scores)], min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
dbscan.fit(X_2d)
labels = dbscan.labels_
labels[labels == -1] = np.max(labels)+1

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = labels, cmap = "tab10")
plt.title('Max. Silh, eps={}, DBSCAN MDS de Mat. Distrib'.format(round(radios[np.argmax(silhouette_scores)],4)))
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(labels)]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_maxsilh_DBSCAN_MDS_distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()


dbscan = DBSCAN(eps=radios[np.argmin(dabo_scores)], min_samples=3)  # eps is the radius for clusters, min_samples is the min points in a cluster
dbscan.fit(X_2d)
labels = dbscan.labels_
labels[labels == -1] = np.max(labels)+1

plt.rcParams.update({'font.size': 44})
plt.figure(figsize=(28, 21))  # Adjust width and height as needed
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], s=400, c = labels, cmap = "tab10")
plt.title('Min. Dabo, eps={} DBSCAN MDS de Mat Distrib'.format(round(radios[np.argmin(dabo_scores)],4)))
# Custom legend with specific text for each cluster
legend_labels = ["Cluster {}".format(cluster+1) for cluster in np.unique(labels)]  # Customize these as you like
# Create legend manually using custom text and colors from the scatter plot
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=15)
                      for i, label in enumerate(legend_labels)]
# Add the legend to the plot
plt.legend(handles=handles, loc="best", ncol=2, framealpha = 0.5)
direccion_guardado = Path("../../../Imagenes/Barrido_final/Distr_encuestas/Clas_mindabo_DBSCAN_MDS_distrib.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close()

"""


func.Tiempo(t0)

