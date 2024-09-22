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
# datos y calcule la distancia de Kolmogorov-Smirnoff 2D. No es exactamente
# eso lo que voy a estar calculando, pero ya veremos nombres después

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
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = 1 - np.min(np.max(Mat_Dist,axis = (1,2)))
    
    ZZ_alterado = np.reshape(ZZ, (ZZ.shape[0]*ZZ.shape[1],ZZ.shape[2]))
    np.savetxt("../Matrices DKS/{}_vs_{}.csv".format(code_y,code_x), ZZ_alterado,delimiter = ",", fmt = "%.6f")

"""

#####################################################################################
#####################################################################################

# Voy a intentar armar una función que levante datos de algún conjunto de
# datos y calcule los cuadrados mínimos entre dos distribuciones como una
# forma de compararlas y construir una métrica entre ambas mediciones.
"""
# Primero levanto los datos de la ANES
Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)
tuplas_preguntas = [('V201372x','V201386x','V200010a'), ('V201408x','V201426x','V200010a'), ('V201372x','V201411x','V200010a')]


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



for preguntas in tuplas_preguntas:
    
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
        Distr_Enc = Distr_Enc.flatten()
    
    # Una pregunta con siete respuestas y otra con seis
    elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
        
        # Saco la cruz
        df_filtered = df_aux[df_aux[code_x] != 4] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[preguntas[2]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:]
        Distr_Enc = Distr_Enc.flatten()
    
    # Dos preguntas con seis respuestas
    elif resp_1 == 6 and resp_2 == 6:
        
        # No hay necesidad de sacar la cruz
        hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_x], y=df_aux[code_y], weights=df_aux[preguntas[2]], vmin=0,cmap = "inferno", density = True,
                  bins=[np.arange(df_aux[code_x].min()-0.5, df_aux[code_x].max()+1.5, 1), np.arange(df_aux[code_y].min()-0.5, df_aux[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d.flatten()
    
    
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
        
        Dist_previa = np.zeros(4)
        
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
            
            for rotacion in range(4):
                
                Dist_previa[rotacion] = np.sum((Distr_Enc-Distr_Sim)**2)
                
                # Una vez que hice el cálculo de la distancia y todo, roto la matriz
                Distr_Sim = np.reshape(Distr_Sim, (6,6))
                Distr_Sim = func.Rotar_matriz(Distr_Sim)
                Distr_Sim = Distr_Sim.flatten()
            
            # Una vez que calcule las 4 distancias habiendo rotado 4 veces la distribución,
            # lo que me queda es guardar eso en las matrices ZZ correspondientes.
            
            repeticion = int(Df_archivos.loc[Df_archivos["nombre"]==nombre,"iteracion"])
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = np.min(Dist_previa)
        
    ZZ_alterado = np.reshape(ZZ, (ZZ.shape[0]*ZZ.shape[1],ZZ.shape[2]))
    np.savetxt("../Matrices DCM/{}_vs_{}.csv".format(code_y,code_x), ZZ_alterado,delimiter = ",", fmt = "%.6f")
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
tuplas_preguntas = [('V201372x','V201386x','V200010a'), ('V201408x','V201426x','V200010a'), ('V201372x','V201411x','V200010a')]

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
        
        df_simil.loc[arc_1,arc_2] = 1- np.min(dist_previa)

df_simil.to_csv("Simil_JS.csv")
"""

#####################################################################################
#####################################################################################

# Ya tengo la matriz de similaridad de JS. A partir de esto puedo fácilmente reconstruir
# la distancia, así que es un poco lo mismo. Me gustaría entonces tomar este archivo y
# levantar cuáles son las distancias inter cluster e intra cluster. Primero para eso
# tengo que definir los clusters.

# Defino las preguntas del cluster de JS
Df_preguntas = pd.read_csv("Tabla_JS.csv")
Clusters = [(0,0.4), (0,0.6), (0.02,1.1), (0.08,1.1), (0.14,1.1), (0.48,0.4)]
preg_cluster = dict()

for tupla in Clusters:
    Cosd = tupla[0]
    Beta = tupla[1]
    
    preg_cluster[tupla] = Df_preguntas.loc[(Df_preguntas["Cosd_100"]==Cosd) & (Df_preguntas["Beta_100"]==Beta), "nombre"]
    
# Levanto los datos de la tabla de similaridad
Df_simil = pd.read_csv("Simil_JS.csv")

for ic1, (tupla_1,archivos_1) in enumerate(preg_cluster.items()):
    
    Promedios = np.zeros(len(archivos_1))
    Varianzas = np.zeros(len(archivos_1))
    
    for grafico_1 in archivos_1:
        
        
    
        for ic2, tupla_2 in enumerate(Clusters):
        
        

        
        
func.Tiempo(t0)

