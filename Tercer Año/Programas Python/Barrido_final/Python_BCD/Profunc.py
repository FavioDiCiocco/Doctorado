#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
"""
# Voy a armar la función que construya los archivos csv con las matrices de distancia
# Jensen-Shannon. La idea es que cada fila del csv tenga uno de los elementos de mi
# matriz, recordando que la matriz es de NxMxP elementos. Entonces cada fila del
# csv debería tener P elementos.

#-------------------------------------------------------------------------------------

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


for preguntas in labels[0:1]:
    
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

# Voy a intentar armar una función que levante datos de algún conjunto de
# datos y calcule la distancia de Kolmogorov-Smirnoff 2D. No es exactamente
# eso lo que voy a estar calculando, pero ya veremos nombres después-

# Primero levanto los datos de una distribución.
Df_ANES, dict_labels = func.Leer_Datos_ANES("../Anes_2020/anes_timeseries_2020.dta", 2020)
preguntas = ('V201372x','V201386x','V200010a')

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
#    Distr_Enc = Distr_Enc.flatten()

# Una pregunta con siete respuestas y otra con seis
elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
    
    # Saco la cruz
    df_filtered = df_aux[df_aux[code_x] != 4] 
    
    hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[preguntas[2]], vmin=0, cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
    plt.close()
    
    Distr_Enc = hist2d[np.arange(7) != 3,:]
#    Distr_Enc = Distr_Enc.flatten()

# Dos preguntas con seis respuestas
elif resp_1 == 6 and resp_2 == 6:
    
    # No hay necesidad de sacar la cruz
    Distr_Enc, xedges, yedges, im = plt.hist2d(x=df_aux[code_x], y=df_aux[code_y], weights=df_aux[preguntas[2]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_aux[code_x].min()-0.5, df_aux[code_x].max()+1.5, 1), np.arange(df_aux[code_y].min()-0.5, df_aux[code_y].max()+1.5, 1)])
    plt.close()
    
#    Distr_Enc = hist2d.flatten()

###############################################################################################################
    
# Lo siguiente que hago es levantar los datos de un archivo

frac_agente_ind = 1/10000
bines = np.linspace(-3.5,3.5,8)
Datos = ldata("../Beta-Cosd/Opiniones_N=10000_kappa=10_beta=0.10_cosd=0.08_Iter=10.file")
dist_final = np.reshape(np.array(Datos[1][:-1],dtype="float"),(42,42))
Opifinales = func.Reconstruccion_opiniones(dist_final, 10000, 2)

Distr_Sim = func.Clasificacion(Opifinales,hist2d.shape[0],hist2d.shape[1],2)
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
# Luego de volver a normalizar mi distribución, si quedaron lugares
# sin agentes, los relleno
restar = np.count_nonzero(Distr_Sim == 0)
Distr_Sim[Distr_Sim == 0] = np.ones(restar)*frac_agente_ind
Distr_Sim[np.argmax(Distr_Sim)] -= frac_agente_ind*restar
Distr_Sim = np.reshape(Distr_Sim, (6,6))

Mat_Dist = np.zeros((6,6,4))
for rotacion in range(4):
    
    Distr_Sim = func.Rotar_matriz(Distr_Sim)
    
    for fila in range(6):
        for columna in range(6):
            
            DKS = np.zeros(4)
            
            # Calculo la distancia en el primer cuadrante (Derecha-arriba)
            DKS[0] = np.sum(Distr_Enc[fila+1:,columna+1:])-np.sum(Distr_Sim[fila+1:,columna+1:])
            
            # Calculo la distancia en el segundo cuadrante (Derecha-abajo)
            DKS[1] = np.sum(Distr_Enc[fila+1:,:columna])-np.sum(Distr_Sim[fila+1:,:columna])
            
            # Calculo la distancia en el tercer cuadrante (Izquierda-abajo)
            DKS[2] = np.sum(Distr_Enc[:fila,:columna])-np.sum(Distr_Sim[:fila,:columna])
            
            # Calculo la distancia en el tercer cuadrante (Izquierda-arriba)
            DKS[3] = np.sum(Distr_Enc[:fila,columna+1:])-np.sum(Distr_Sim[:fila,columna+1:])
            
            Mat_Dist[fila,columna,rotacion] = np.max(DKS)




func.Tiempo(t0)
