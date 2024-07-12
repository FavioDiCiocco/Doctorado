# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:33:00 2022

@author: Favio
"""

# Este archivo es para definir funciones

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np
import time
import math
from pathlib import Path

##################################################################################
##################################################################################

# FUNCIONES GENERALES

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------

# Esto printea una cantidad de valores cant de un objeto iterable que paso
# en la parte de lista.
def scan(lista,cant=10):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
#--------------------------------------------------------------------------------
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo(t0):
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))

#--------------------------------------------------------------------------------

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

#--------------------------------------------------------------------------------    
    
def data_processor(x):
    if(isinstance(x, int)):
        return x
    elif(isinstance(x, float)):
        return int(x)
    elif(isinstance(x, str)):
        return int(x[0]) if(x[0]!="-" and int(x[0])<9) else 0
    elif(x.isnan()):
        return 0
    else:
        print("Error, no se ha identificado el tipo: {}".format(type(x)))

##################################################################################
##################################################################################

# FUNCIONES GRAFICADORAS

##################################################################################
##################################################################################

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Graf_Histograma_opiniones_2D(DF,path,carpeta,bins,cmap,
                                 ID_param_x,ID_param_y,ID_param_extra_1):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_x = Arr_param_x[(Arr_param_x >= 0) & (Arr_param_x <= 0.04)]
    
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_param_y = Arr_param_y[(Arr_param_y >= 0.4) & (Arr_param_y <= 0.8)]
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    # Diccionario con la entropía, Sigma_x, Sigma_y, Promedios y Covarianzas
    # de todas las simulaciones para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path, 20, 20)
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y in Tupla_total:
            
            Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            for nombre in archivos:
                
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                if repeticion < 10:
                
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    # Levanto los datos del archivo
                    Datos = ldata(path / nombre)
                    
                    # Leo los datos de las Opiniones Finales
                    Opifinales = np.array(Datos[5][:-1:], dtype="float")
                    Opifinales = (Opifinales/EXTRAS)*bins[-1]
                    X_0 = Opifinales[0::T]
                    Y_0 = Opifinales[1::T]
                    
                    # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    
                    # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                
                    # direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                                                # ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS,repeticion))
                    
                    direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,
                                                                                                ID_param_x,PARAM_X,ID_param_y,PARAM_Y,repeticion))
                    
                    indice = np.where(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] == repeticion)[0][0]
                    estado = int(Frecuencias[indice])
                    
                    Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
                               "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
                               "Polarización 1D y Consenso con anchura",
                               "Polarización Ideológica con anchura", "Transición con anchura",
                               "Polarización Descorrelacionada con anchura"]
                    
                    X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                    Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                    
                    # Armo mi gráfico, lo guardo y lo cierro
                    
                    plt.rcParams.update({'font.size': 32})
                    plt.figure(figsize=(20,15))
                    # _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                    #                          range=[[-EXTRAS,EXTRAS],[-EXTRAS,EXTRAS]],density=True,
                    #                          cmap=cmap)
                    _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="inferno")
                    plt.xlabel(r"$x_i^1$")
                    plt.ylabel(r"$x_i^2$")
                    plt.title('Histograma 2D, {}={:.2f}_{}={:.2f}\n{}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y,Nombres[estado]))
                    plt.colorbar(im, label='Fracción')
                    plt.savefig(direccion_guardado ,bbox_inches = "tight")
                    plt.close()

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Diccionario_metricas(DF, path, Nx, Ny):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    Salida = dict()
    for EXTRAS in Arr_EXTRAS:
        Salida[EXTRAS] = dict()
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            Varianza_X = np.zeros(archivos.shape[0])
            Varianza_Y = np.zeros(archivos.shape[0])
            Covarianza = np.zeros(archivos.shape[0])
            Promedios = np.zeros(archivos.shape[0])
            Entropia = np.zeros(archivos.shape[0])
            Identidad = np.zeros(archivos.shape[0], dtype=int)
            
            for indice,nombre in enumerate(archivos):
                
        
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Semilla
        
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
        
                # Leo los datos de las Opiniones Finales
                Opifinales = np.zeros((T,AGENTES))
        
                for topico in range(T):
                    Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")
                    Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                Identidad[indice] = repeticion
        
                M_cov = np.cov(Opifinales, bias = True)
                Varianza_X[indice] = M_cov[0,0]
                Varianza_Y[indice] = M_cov[1,1]
                Covarianza[indice] = M_cov[0,1]
                Promedios[indice] = np.linalg.norm(np.array(Datos[5][:-1:], dtype="float"),ord=1) / np.array(Datos[5][:-1:], dtype="float").shape[0]
                
                # Tengo que rearmar Opifinales para que sea un sólo vector con todo
                
                Opifinales = np.array(Datos[5][:-1], dtype="float")
                Opifinales = Opifinales/EXTRAS
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,Nx, Ny,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                Entropia[indice] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
                
            #----------------------------------------------------------------------------------------------------------------------
            
            # Mis datos no están ordenados, pero con esto los ordeno según el
            # valor de la simulación. Primero inicializo el vector que tiene los índices
            # de cada simulación en sus elementos. El elemento 0 tiene la ubicación
            # de la simulación cero en los demás vectores.
            Ubicacion = np.zeros(max(Identidad)+1,dtype = int)
            
            # Para cada elemento en el vector de identidad, le busco su indice en el
            # vector y coloco ese índice en el vector de Ubicacion en la posición
            # del elemento observado
            for i in np.sort(Identidad):
                indice = np.where(Identidad == i)[0][0]
                Ubicacion[i] = indice
            
            # Ahora tengo que remover las simulaciones faltantes. Armo un vector
            # que tenga sólamente los índices de las simulaciones faltantes
            Faltantes = np.arange(max(Identidad)+1)
            Faltantes = np.delete(Faltantes,Identidad)
            
            # Borro esas simulaciones de mi vector de Ubicacion
            Ubicacion = np.delete(Ubicacion,Faltantes)
                
                
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            if PARAM_Y not in Salida[EXTRAS][PARAM_X].keys():
                Salida[EXTRAS][PARAM_X][PARAM_Y] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Entropia"] = Entropia[Ubicacion] / np.log2(Nx*Ny)
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"] = Varianza_X[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"] = Varianza_Y[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"] = Covarianza[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Promedios"] = Promedios[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] = np.sort(Identidad)
            
    return Salida

#-----------------------------------------------------------------------------------------------

def Identificacion_Estados(Entropia, Sigma_X, Sigma_Y, Covarianza, Promedios):
    
    Resultados = np.zeros(len(Entropia))
    
    for i,ent,sx,sy,cov,prom in zip(np.arange(len(Entropia)),
                                    Entropia, Sigma_X, Sigma_Y, Covarianza, Promedios):
        
        # Reviso la entropía y separo en casos con y sin anchura
        
        if ent <= 0.3:
            
            # Estos son casos sin anchura
            
            if sx < 0.5 and sy < 0.5:
                
                # Caso de un sólo extremo
                
                # Consenso neutral
                if prom < 0.1:
                    Resultados[i] = 0
                
                # Consenso radicalizado
                else:
                    Resultados[i] = 1
                    
            
            # Casos de dos extremos
            elif sx >= 0.5 and sy < 0.5:
                # Dos extremos horizontal
                Resultados[i] = 2
            elif sx < 0.5 and sy >= 0.5:
                # Dos extremos vertical
                Resultados[i] = 2
                
            else:
                if np.abs(cov) > 0.6:
                    # Dos extremos ideológico
                    Resultados[i] = 3
                elif np.abs(cov) < 0.3:
                    # Cuatro extremos
                    Resultados[i] = 5
                else:
                    # Estados de Transición
                    Resultados[i] = 4
        
        else:
            
            # Estos son los casos con anchura
            
            # Casos de dos extremos
            if sx >= 0.3 and sy < 0.3:
                # Dos extremos horizontal
                Resultados[i] = 6
            elif sx < 0.3 and sy >= 0.3:
                # Dos extremos vertical
                Resultados[i] = 6
            
            else:
                # Polarización
                # Polarización ideológica
                if np.abs(cov) >= 0.25:
                    Resultados[i] = 7
                
                # Transición con anchura
                elif np.abs(cov) >= 0.15 and np.abs(cov) < 0.25:
                    Resultados[i] = 8
                
                # Polarización descorrelacionada
                else:
                    Resultados[i] = 9
                
    return Resultados


#-----------------------------------------------------------------------------------------------

def Clasificacion(Array, Nx, Ny,T):
    
    # Recibo un array de opiniones que van entre [-1,1]. Le sumo 1
    # para que las opiniones vayan entre [0,2].
    Array = Array+1
    
    # Divido mi espacio de tópicos 2D en cuadrados. Defino el ancho
    # de esos cuadrados.
    ancho_x = 2/Nx
    ancho_y = 2/Ny
    
    # Armo un array de tuplas que indiquen "fila" y "columna" en la cuál
    # cae cada opinión.
    Ubicaciones = np.array([(math.floor(x/ancho_x),math.floor(y/ancho_y)) for x,y in zip(Array[0::T],Array[1::T])])
    
    # Ahora me armo mi array de distribución, que cuenta cuántas opiniones tengo
    # por cada cajita.
    Distribucion = np.zeros((Nx*Ny))
    for opinion in Ubicaciones:
        # Tomo mínimos para que no intente ir a una cajita no existente. Tendría un problema
        # si algún agente tiene opinión máxima en algún tópico.
        fila = min(opinion[0],Nx-1)
        columna = min(opinion[1],Ny-1)
        Distribucion[fila*Ny+columna] += 1
    
    # Una vez armada mi distribucion, la normalizo.
    Distribucion = Distribucion/np.sum(Distribucion)
    
    # Returneo la distribucion
    return Distribucion

#-----------------------------------------------------------------------------------------------

# Con la siguiente función me leo los datos de la ANES

def Leer_Datos_ANES(filename,año):
    
    # Leo los datos del archivo que tiene formato dta
    df_raw_data = pd.read_stata(filename)
    
    # Después tendría que armar una separación en casos a partir del nombre
    # de la ANES, no de un número externo
    if año == 2020:
        # Brief description of the codes
        dict_labels = {'V201200':'Liberal-Conservative self Placement', 'V201225x':'Voting as duty or choice','V201231x':'Party Identity',
                       'V201246':'Spending & Services', 'V201249':'Defense Spending', 'V201252':'Gov-private Medical Insurance',
                       'V201255':'Guaranteed job Income', 'V201258':'Gov Assistance to Blacks', 'V201262':'Environment-Business Tradeoff',
                       'V201342x':'Abortion Rights Supreme Court', 'V201345x':'Death Penalty','V201356x':'Vote by mail',
                       'V201362x':'Allowing Felons to vote', 'V201372x':'Pres didnt worry Congress',
                       'V201375x':'Restricting Journalist access', 'V201382x':'Corruption increased or decreased since Trump',
                       'V201386x':'Impeachment', 'V201405x':'Require employers to offer paid leave to parents',
                       'V201408x':'Service same sex couples', 'V201411x':'Transgender Policy', 'V201420x':'Birthright Citizenship',
                       'V201423x':'Should children brought illegally be sent back','V201426x':'Wall with Mexico',
                       'V201429':'Urban Unrest','V201605x':'Political Violence compared to 4 years ago',
                       'V202236x':'Allowing refugees to come to US','V202239x':'Effect of Illegal inmigration on crime rate',
                       'V202242x':'Providing path to citizenship','V202245x':'Returning unauthorized immigrants to native country',
                       'V202248x':'Separating children from detained immigrants','V202255x':'Less or more Government',
                       'V202256':'Good for society to have more government regulation',
                       'V202259x':'Government trying to reduce income inequality','V202276x':'People in rural areas get more/less from Govt.',
                       'V202279x':'People in rural areas have too much/too little influence','V202282x':'People in rural areas get too much/too little respect',
                       'V202286x':'Easier/Harder for working mother to bond with child','V202290x':'Better/Worse if man works and woman takes care of home',
                       'V202320x':'Economic Mobility','V202328x':'Obamacare','V202331x':'Vaccines Schools',
                       'V202336x':'Regulate Greenhouse Emissions','V202341x':'Background checks',
                       'V202344x':'Banning Rifles','V202347x':'Government buy back of "Assault-Style" Rifles',
                       'V202350x':'Govt action about opiods','V202361x':'Free trade agreements with other countries',
                       'V202376x':'Federal program giving 12K a year to citizens','V202380x':'Government spending to help pay for health care',
                       'V202383x':'Benefits of vaccination','V202390x':'Trasgender people serve in military',
                       'V202490x':'Government treats whites or blacks better','V202493x':'Police treats whites or blacks better',
                       'V202542':'Use Facebook','V202544':'Use Twitter'}

        labels = list(dict_labels.keys())

        labels_pre = list()
        labels_post = list()

        for label in labels:
            if label[3] == "1":
                labels_pre.append(label)
            elif label[3] == "2":
                labels_post.append(label)
                
        # Primer Filtro

        labels_politicos = ['V201200','V201231x','V201372x','V201386x','V201408x',
                            'V201411x','V201420x','V201426x','V202255x','V202328x','V202336x']

        labels_apoliticos = ['V201429','V202320x','V202331x','V202341x','V202344x',
                             'V202350x','V202383x']

        labels_dudosos = ['V201225x','V201246','V201249','V201252','V201255','V201258',
                          'V201262','V202242x','V202248x']

        labels_filtrados = labels_politicos + labels_apoliticos + labels_dudosos
        
        # Preparo el dataframe que voy a sacar
        df_data_aux = df_raw_data[labels]
        df_data = pd.DataFrame()

        for code in labels_filtrados:
            df_data[code] = df_data_aux[code].apply(data_processor)
        
        # Asigno los pesos
        df_data[['V200010a','V200010b']] = df_raw_data[['V200010a','V200010b']]

    return df_data, dict_labels

#-----------------------------------------------------------------------------------------------
    
# Esta función construye la matriz de DJS y la returnea
    
def Matriz_DJS(DF_datos,DF_Anes,Dic_ANES,path):
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    frac_agente_ind = 1/AGENTES
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((XX.shape[0],XX.shape[1],100))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo mis distribuciones sacando la cruz, que sería sacar a los agentes que
    # respondieron neutro en alguna encuesta.
    
    # Separo las opiniones de 0
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    # Reviso la cantidad de respuestas de cada pregunta
    resp_1 = np.unique(df_aux[Dic_ANES["code_1"]]).shape[0]
    resp_2 = np.unique(df_aux[Dic_ANES["code_2"]]).shape[0]
    
    # Los clasifico como código x y código y
    if resp_1 >= resp_2:
        code_x = Dic_ANES["code_1"]
        code_y = Dic_ANES["code_2"]
    else:
        code_x = Dic_ANES["code_2"]
        code_y = Dic_ANES["code_1"]
    
    
    # Dos preguntas con siete respuestas
    if resp_1 == 7 and resp_2 == 7:
        
        # Saco la cruz
        df_filtered = df_aux[(df_aux[code_x] != 4) & (df_aux[code_y] != 4)] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[Dic_ANES["weights"]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:][:,np.arange(7) != 3]
        Distr_Enc = Distr_Enc.flatten()
    
    # Una pregunta con siete respuestas y otra con seis
    elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
        
        # Saco la cruz
        df_filtered = df_aux[df_aux[code_x] != 4] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[Dic_ANES["weights"]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:]
        Distr_Enc = Distr_Enc.flatten()
    
    # Dos preguntas con seis respuestas
    elif resp_1 == 6 and resp_2 == 6:
        
        # No hay necesidad de sacar la cruz
        hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_x], y=df_aux[code_y], weights=df_aux[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
                  bins=[np.arange(df_aux[code_x].min()-0.5, df_aux[code_x].max()+1.5, 1), np.arange(df_aux[code_y].min()-0.5, df_aux[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d.flatten()
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        
        #-----------------------------------------------------------------------------------------
        
        Dist_previa = np.zeros(4)
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Pasos simulados
            # Semilla
            # Matriz de Adyacencia
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales y me armo una distribución en forma de matriz de 7x7
            Opifinales = np.array(Datos[5][:-1], dtype="float")
            Opifinales = Opifinales / EXTRAS
            Distr_Sim = Clasificacion(Opifinales,hist2d.shape[0],hist2d.shape[1],T)
            
            # La Distr_Orig es un array plano. Lo que estaria bueno es armarlo de 6x6,
            # total la Distr_Enc tiene tamaño 6x6 siempre
            
            Distr_Sim = np.reshape(Distr_Sim, hist2d.shape)
            
            # Dos preguntas con siete respuestas
            if resp_1 == 7 and resp_2 == 7:
                Distr_Sim = Distr_Sim[np.arange(7) != 3,:][:,np.arange(7) != 3]
            
            # Una pregunta con siete respuestas y otra con seis
            elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
                Distr_Sim = Distr_Sim[np.arange(7) != 3,:]
            
            # Dos preguntas con seis respuestas (En este caso no necesita hacer nada, ya es de 6x6 la distribución)
            
            
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
            
            #-----------------------------------------------------------------------------------------
            
            # Calculo la distancia Jensen-Shannon de la distribución de la encuesta con la distribución
            # de la simulación. Hago esto cuatro veces porque tengo que rotar cuatro veces la distribución
            # simulada, cosa de asegurarme de que no importe la dirección del estado final.
            
            for rotacion in range(4):
                
                Dist_previa[rotacion] = jensenshannon(Distr_Enc,Distr_Sim)
                
                # Una vez que hice el cálculo de la distancia y todo, roto la matriz
                Distr_Sim = np.reshape(Distr_Sim, (6,6))
                Distr_Sim = Rotar_matriz(Distr_Sim)
                Distr_Sim = Distr_Sim.flatten()
                
            #-----------------------------------------------------------------------------------------
            
            # Una vez que calcule las 4 distancias habiendo rotado 4 veces la distribución,
            # lo que me queda es guardar eso en las matrices ZZ correspondientes.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = np.min(Dist_previa)
            
            #-----------------------------------------------------------------------------------------
            
    # Resuelta la matriz, returneo mi info
    
    return ZZ, code_x, code_y

#-----------------------------------------------------------------------------------------------

# Armo dos mapas de colores de DJS. El mapa de colores
# no tiene los agentes que hayan opinado neutro en ninguna de las preguntas.
# En ambos casos estoy considerando que ambas preguntas tienen 7 respuestas. Voy a tener que ir
# viendo cómo resolver si tienen 6 respuestas.

def Mapas_Colores_DJS(Dist_JS, code_x, code_y, DF_datos, Dic_ANES, dict_labels, carpeta,
                      ID_param_x,SIM_param_x,ID_param_y,SIM_param_y):
    
    # Defino los arrays de parámetros diferentes
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    # Organizo las matrices Dist_JS según su similaridad
    Dist_JS = np.sort(Dist_JS)
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores para el caso sin cruz
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_{}vs{}.png".format(carpeta,code_y,code_x))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Distancia Jensen-Shannon",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,np.mean(Dist_JS, axis=2),shading="nearest", cmap = "viridis")
    tupla = np.unravel_index(np.argmin(np.mean(Dist_JS,axis=2)),np.mean(Dist_JS,axis=2).shape)
    plt.colorbar()
    plt.scatter(XX[tupla],YY[tupla], marker="X", s = 1500, color = "red")
    plt.title("Distancia Jensen-Shannon\n {} vs {}".format(dict_labels[code_y],dict_labels[code_x]))
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Distancia Jensen-Shannon")
    
    # Y ahora me armo el gráfico de promedios de distancia JS según cantidad de simulaciones
    # consideradas, con las simulaciones ordenadas de las de menos distancia a las de más distancia
    """
    for i in range(3):
        
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_{}vs{}_r{}.png".format(carpeta,code_y,code_x,i))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("Ranking Distancia Jensen-Shannon",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        Dist_JS_prom = np.mean(Dist_JS[:,:,0:10+i*20],axis=2)
        
        # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
        tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
        
        plt.pcolormesh(XX,YY,Dist_JS_prom,shading="nearest", cmap = "cividis")
        plt.colorbar()
        plt.scatter(XX[tupla],YY[tupla], marker="X", s = 1500, color = "red")
        
        plt.title("Distancia Jensen-Shannon {} simulaciones\n {} vs {}".format(10+i*20,dict_labels[code_y],dict_labels[code_x]))
        
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Ranking Distancia Jensen-Shannon")
        
    """
    

#-----------------------------------------------------------------------------------------------

# Esta función arma los histogramas de opiniones máxima y mínima similaridad entre las 10 simulaciones
# más similares con la distribución de la encuesta

def Hist2D_similares_FEF(Dist_JS, code_x, code_y, DF_datos, Dic_Total, Dic_ANES, dict_labels, carpeta, path, bins,
                         SIM_param_x,SIM_param_y):
    
    # Hago los gráficos de histograma 2D de las simulaciones que más se parecen y que menos se parecen
    # a mis distribuciones de las encuestas
    # Dist_JS_sorted = np.sort(Dist_JS)
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    
    #-------------------------------------------------------------------------------------------------
    
    # Antes de ordenar mis matrices, voy a obtener la info del gráfico que más se parece
    # y del décimo que más se parece se parece a lo que estoy queriendo comparar.
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    iMin = np.argsort(Dist_JS[tupla])[0]
    
    # Hallo el décimo que más se parece a la distribución. Arranco con el que no tiene centro
    
#    flattened_array = Dist_JS.flatten()
#    sorted_indices = np.argsort(flattened_array)
#    tenth_element_flat_index = sorted_indices[9]
#    iMax = np.unravel_index(tenth_element_flat_index, Dist_JS.shape)
    
    #--------------------------------------------------------------------------------

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    T = 2
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
#    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
#                   for j,param_y in enumerate(Arr_param_y)]
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    #--------------------------------------------------------------------------------
    
    # Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
    #        "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
    #        "Polarización 1D y Consenso con anchura",
    #        "Polarización Ideológica con anchura", "Transición con anchura",
    #        "Polarización Descorrelacionada con anchura"]
    
    simil = "min_distancia"
    distan = np.min(Dist_JS[tupla])
    
    #--------------------------------------------------------------------------------
    # for tupla,simil,distan in zip([iMin, iMax],Lista_similaridad,Valor_distancia):
    
    # Armo listas de strings y números para mis archivos
#    Lista_similaridad = ["min_distancia","max_distancia"]
#    Valor_distancia = [np.min(Dist_JS_sorted),np.max(Dist_JS_sorted[:,:,0:10])]
    #--------------------------------------------------------------------------------
    
    PARAM_X = XX[tupla[0],tupla[1]]
    PARAM_Y = YY[tupla[0],tupla[1]]
    
    # Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
    #                                              Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
    #                                              Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
    #                                              Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
    #                                              Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
    
    # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
    archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                (DF_datos["n"]==AGENTES) & 
                                (DF_datos["Extra"]==EXTRAS) & 
                                (DF_datos["parametro_x"]==PARAM_X) &
                                (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
    
    #-----------------------------------------------------------------------------------------
    
    for nombre in archivos:
        
        repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
        if repeticion == iMin:
        
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.array(Datos[5][:-1:], dtype="float")
            Opifinales = (Opifinales/EXTRAS)*bins[-1]
            X_0 = Opifinales[0::T]
            Y_0 = Opifinales[1::T]
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
            
            direccion_guardado = Path("../../../Imagenes/{}/Hist_2D_{}_{}vs{}.png".format(carpeta /"Sin Cruz",simil,code_y,code_x))
            
            # indice = np.where(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] == repeticion)[0][0]
            # estado = int(Frecuencias[indice])
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Tengo que armar los valores de X e Y que voy a graficar
            
            X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
            Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
            
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Armo mi gráfico, lo guardo y lo cierro
            
            plt.rcParams.update({'font.size': 44})
            plt.figure(figsize=(28,21))
            _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="inferno")
            plt.xlabel(r"$x_i^1$")
            plt.ylabel(r"$x_i^2$")
            plt.title(r'Distancia JS = {:.2f}, ${}$={:.2f}, ${}$={:.2f}'.format(distan,SIM_param_x,PARAM_X,SIM_param_y,PARAM_Y) + '\n {} vs {}'.format(dict_labels[code_y],dict_labels[code_x]))
            plt.colorbar(im, label='Fracción')
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close()
             
    
    #--------------------------------------------------------------------------------
    """
    # Lo que quiero hacer acá es armar gráficos de promedios de distancias rankeados.
    
    for i in range(10):
        
        # Hago el ploteo del mapa de colores con el colormesh
        Dist_JS_prom = np.mean(Dist_JS_sorted[:,:,0:10+i*10],axis=2)
        # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
        tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
        
        PARAM_X = XX[tupla[0],tupla[1]]
        PARAM_Y = YY[tupla[0],tupla[1]]
        
    #-----------------------------------------------------------------------------------------
    
        OpiTotales = np.empty(0)
        cant_simulaciones = 10+i*10
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        
        i_Proms = np.argsort(Dist_JS[tupla])[0:cant_simulaciones]
        
        for nombre in archivos[i_Proms]:
        
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.array(Datos[5][:-1:], dtype="float")
            Opifinales = (Opifinales/EXTRAS)*bins[-1]
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            OpiTotales = np.concatenate((OpiTotales,Opifinales),axis=0)
            
        X_0 = OpiTotales[0::T]
        Y_0 = OpiTotales[1::T]
        
        # Tengo que armar los valores de X e Y que voy a graficar
        
        X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
        Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
        
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/Hists_prom_{}vs{}_r{}.png".format(carpeta,code_y,code_x,i))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("Ranking Opiniones Promedio",figsize=(28,21))
        _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="inferno")
        plt.xlabel(r"$x_i^1$")
        plt.ylabel(r"$x_i^2$")
        plt.title(r'Promedio de Histogramas, {} simulaciones, ${}$={}, ${}$={}'.format(cant_simulaciones,SIM_param_x,PARAM_X,SIM_param_y,PARAM_Y) +'\n {} vs {}'.format(dict_labels[code_y],dict_labels[code_x]))
        plt.colorbar(im, label='Fracción')
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close("Ranking Opiniones Promedio")
        
    
    #-------------------------------------------------------------------------------------------------
    
    
    # Lo que quiero hacer acá es armar gráficos de frecuencia de estados de los dos
    # estados más probables en el espacio de parámetros.
    
    # Arranco con un barrido en el ranking
    for i in range(10):
        
        cant_simulaciones = 10+i*10
        # Me construyo la matriz en la que voy a guardar los estados clasificados
        Estados_clasificados = np.zeros((XX.shape[0],XX.shape[1],cant_simulaciones))
        
        # Luego hago un barrido en los parámetros
        
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
            
            #-----------------------------------------------------------------------------------------
            
            # Reviso los vectores de las distancias Jensen Shannon ya calculados y obtengo las posiciones de
            # las simulaciones cuyas distancias sean menores.
            iElemR = np.argsort(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,:])[cant_simulaciones-1]
#            if PARAM_X == 0.1:
#                print("Cos(delta): ",PARAM_X)
#                print("Beta: ",PARAM_Y)
#                print("indice: ",iElemR)
#                print("Cant True: ",np.count_nonzero(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]))
#                print("Tamaño vector Booleano: ",len(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]))
#                print("Cant Frecuencias: ",len(Frecuencias[Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]]))
#                print(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna])
            Array_bool = Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]
            Estados_clasificados[(Arr_param_y.shape[0]-1)-fila,columna, :] = Frecuencias[Array_bool][0:cant_simulaciones]
            
        # Ahora que tengo los estados clasificados de las simulaciones dentro del conjunto de simulaciones
        # más similares, lo siguiente es determinar cuáles son los estados predominantes, así hago gráficos
        # de los dos estados más predominantes.
        
        Estados, Cuentas = np.unique(Estados_clasificados, return_counts=True)
        estados_dominantes = Estados[np.argsort(Cuentas)[-2:]]
        
        # Construyo mi array ZZ para graficar el mapa de colores de FEF.
        ZZ_estados = np.zeros((2,XX.shape[0],XX.shape[1]))
        
        for grafico,estado in enumerate(estados_dominantes):
            for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
                
                ZZ_estados[grafico,(Arr_param_y.shape[0]-1)-fila,columna] = np.count_nonzero(Estados_clasificados[(Arr_param_y.shape[0]-1)-fila,columna] == estado)/Estados_clasificados.shape[2]
                
        for grafico,titulo,direc in zip(np.arange(2),["Segundo estado más probable","Primer estado más probable"],["SegundoDom","PrimeroDom"]):
            # Una vez que tengo el ZZ completo, armo mi mapa de colores
            direccion_guardado = Path("../../../Imagenes/{}/FEF {}_{}vs{}_r{}.png".format(carpeta/"Sin Cruz",direc,code_y,code_x,i))
            
            plt.rcParams.update({'font.size': 44})
            plt.figure("FEF",figsize=(28,21))
            plt.xlabel(r"${}$".format(SIM_param_x))
            plt.ylabel(r"${}$".format(SIM_param_y))
            
            # Hago el ploteo del mapa de colores con el colormesh

            plt.pcolormesh(XX,YY,ZZ_estados[grafico],shading="nearest", cmap = "plasma",vmin = 0, vmax = 1)
            plt.colorbar()
            plt.title("Fracción de estados de {} \n {}, {} simulaciones \n {} vs {}".format(Nombres[int(estados_dominantes[grafico])],titulo,cant_simulaciones,dict_labels[code_y],dict_labels[code_x]))
            
            # Guardo la figura y la cierro
            
            plt.savefig(direccion_guardado , bbox_inches = "tight")
            plt.close("FEF")
    """

#-----------------------------------------------------------------------------------------------

# Armo una función que en el punto de mínima distancia media construya un histograma de las distancias de JS

def Histograma_distancias(Dist_JS, code_x, code_y, DF_datos, dict_labels, carpeta,
                          ID_param_x, SIM_param_x, ID_param_y, SIM_param_y):
    
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    #-----------------------------------------------------------------------------------------
    
    # Lo que quiero hacer acá es armar gráficos de promedios de opiniones rankeados.
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    bines = np.linspace(0,1,41)
    
#    for tupla in lminimos:
    
    barrX = XX[max(tupla[0]-1,0):tupla[0]+2,max(tupla[1]-1,0):tupla[1]+2].flatten()
    barrY = YY[max(tupla[0]-1,0):tupla[0]+2,max(tupla[1]-1,0):tupla[1]+2].flatten()
    Distancias = np.reshape(Dist_JS[max(tupla[0]-1,0):tupla[0]+2,max(tupla[1]-1,0):tupla[1]+2],(barrX.shape[0],Dist_JS.shape[2]))
    
    for PARAM_X,PARAM_Y,Arr_Dist in zip(barrX,barrY,Distancias):
        Y, _ = np.histogram(Arr_Dist, bins = bines)
        
        # Set the figure size
        plt.rcParams.update({'font.size': 44})
        plt.figure(figsize=(28, 21))  # Adjust width and height as needed
        plt.bar(bines[:-1], Y/np.sum(Y), width = (bines[1]-bines[0])*0.9, align = "edge")
        plt.xlabel("Distancia JS")
        plt.ylabel("Probabilidad")
        plt.xlim(bines[:-1][Y>0][0]-0.025,bines[:-1][Y>0][-1]+0.075)
#        plt.axvline(x=0.45, linestyle = "--", color = "red", linewidth = 4)
        plt.title("{} vs {}\n".format(dict_labels[code_y],dict_labels[code_x]) + r"${}$={}, ${}$={}".format(SIM_param_y,PARAM_Y,SIM_param_x,PARAM_X))
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/Hist distancias_{} vs {}_{}={}_{}={}.png".format(carpeta,code_y,code_x,ID_param_y,PARAM_Y,ID_param_x,PARAM_X))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()
    

#-----------------------------------------------------------------------------------------------

# Lo que quiero es ver cuál es la composición de los estados que son parte del cluster
# de distancias pequeñas que observo en el histograma de Distancias. 

def Comp_estados(Dist_JS, code_x, code_y, DF_datos, Dic_Total, dict_labels, carpeta, path, dist_lim,
                 ID_param_x, SIM_param_x, ID_param_y, SIM_param_y):
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    
#    for imin,tupla in enumerate(lminimos):
    
    # cant_sim = np.count_nonzero(Dist_JS[tupla] <= dist_lim)
    
    # for PARAM_X,PARAM_Y in zip(XX[tupla[0]-1:tupla[0]+2,tupla[1]-1:tupla[1]+2].flatten(),YY[tupla[0]-1:tupla[0]+2,tupla[1]-1:tupla[1]+2].flatten()):
    
    Nombres = ["Cons Neut", "Cons Rad", "Pol 1D y Cons","Pol Id", "Trans", "Pol Desc","Pol 1D y Cons anch","Pol Id anch", "Trans anch","Pol Desc anch"]
    
    bin_F = np.arange(-0.5,10.5)
    bin_D = np.linspace(0,1,41)
    X = np.arange(10)
    
    barrX = XX[max(tupla[0]-1,0):tupla[0]+2,max(tupla[1]-1,0):tupla[1]+2].flatten()
    barrY = YY[max(tupla[0]-1,0):tupla[0]+2,max(tupla[1]-1,0):tupla[1]+2].flatten()
    Distancias = np.reshape(Dist_JS[max(tupla[0]-1,0):tupla[0]+2,max(tupla[1]-1,0):tupla[1]+2],(barrX.shape[0],Dist_JS.shape[2]))
    
    for j,PARAM_X,PARAM_Y,Arr_Dist in zip(np.arange(barrX.shape[0]),barrX,barrY,Distancias):
        
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
        
        
        for i,dmin,dmax in zip(np.arange(bin_D.shape[0]-1),bin_D[0:-1],bin_D[1:]):
            Arr_bool = (Arr_Dist >= dmin) & (Arr_Dist <= dmax) 
            if np.count_nonzero(Arr_bool) == 0:
                continue
            plt.rcParams.update({'font.size': 44})
            plt.figure(figsize=(28, 21))  # Adjust width and height as needed
            plt.hist(Frecuencias[Arr_bool], bins = bin_F, density = True)
            plt.ylabel("Fracción")
            plt.title('{} vs {}\n'.format(dict_labels[code_y], dict_labels[code_x]) + r'Cantidad simulaciones {}, ${}$={},${}$={}, Distancias entre {:.2f} y {:.2f}'.format(np.count_nonzero(Arr_bool), SIM_param_y, PARAM_Y, SIM_param_x, PARAM_X, dmin, dmax))
            plt.xticks(ticks = X, labels = Nombres, rotation = 45)
            direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/Comp est_{}vs{}_min={}_b{}.png".format(carpeta,code_y,code_x,j,i))
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close()
    
    #-----------------------------------------------------------------------------------------
        
    # Una vez que tengo el ZZ completo, armo mi mapa de colores para el caso sin cruz
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_recortado_{}vs{}.png".format(carpeta,code_y,code_x))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Distancia Jensen-Shannon",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    cant_sim = np.count_nonzero(Dist_JS[tupla] <= dist_lim)
    
    # Hago el ploteo del mapa de colores con el colormesh
    Dist_JS_prom = np.mean(np.sort(Dist_JS)[:,:,0:cant_sim],axis=2)
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,Dist_JS_prom,shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.scatter(XX[tupla],YY[tupla], marker="X", color = "red", s = 1500)
    plt.title("Distancia Jensen-Shannon \n {} vs {}\nCantidad simulaciones {}".format(dict_labels[code_y],dict_labels[code_x], cant_sim))
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Distancia Jensen-Shannon")
    
#-----------------------------------------------------------------------------------------------

# A partir de mis gráficos de histogramas de distancias, quiero armar un gráfico
# que mire los histogramas y construya la cantidad de gráficos que tienen X configuraciones
# con distancias menor a la distancia límite, que por ahora es 0.45.

def FracHist_CantEstados(Dist_JS, code_x, code_y, DF_datos, dict_labels, carpeta, path, dist_lim):
    
    # Primero reviso mis conjuntos de distancias para cada punto del espacio. En esos puntos,
    # cuento cuántas de las simulaciones tienen distancia menor a la distancia de corte
    Cantidad = np.zeros(Dist_JS.shape[0]*Dist_JS.shape[1])
    for indice,distancias in enumerate(np.reshape(Dist_JS,(Dist_JS.shape[0]*Dist_JS.shape[1],Dist_JS.shape[2]))):
        Cantidad[indice] = np.count_nonzero(distancias <= dist_lim)
    
    # Con eso puedo entonces contar cuántos histogramas tienen X simulaciones con distancias
    # menor a la distancia de corte.
    Y = np.zeros(Dist_JS.shape[2]+1)
    unicos,cant = np.unique(Cantidad, return_counts = True)
    Y[unicos.astype(int)] = cant/np.sum(cant)
    
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    cant_sim = np.count_nonzero(Dist_JS[tupla] <= dist_lim)
    
    # Set the figure size
    plt.rcParams.update({'font.size': 44})
    plt.figure(figsize=(28, 21))  # Adjust width and height as needed
    plt.plot(np.arange(Dist_JS.shape[2]+1), Y, "--g", linewidth = 6)
    plt.xlabel("Número de configuraciones con distancia menor a {}".format(dist_lim))
    plt.ylabel("Fracción de Histogramas")
    plt.axvline(x=cant_sim, linestyle = "--", color = "red", linewidth = 4)
    plt.title("{} vs {}".format(dict_labels[code_y],dict_labels[code_x]))
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/FracHistvsEstados_{} vs {}.png".format(carpeta,code_y,code_x))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()

#-----------------------------------------------------------------------------------------------

# Voy a realizar dos mapas de colores. Uno con el promedio de un subconjunto de distancias
# de JS y otro con la fracción de simulaciones que estoy contando para ese promedio

def Doble_Mapacol_PromyFrac(Dist_JS, code_x, code_y, DF_datos, dict_labels, carpeta, path,
                            SIM_param_x, SIM_param_y):
    
    # Arranco eligiendo un rango de criterios. Yo diría de mover entre 0.25 y 0.75
    Criterios = np.arange(6)*0.1 + 0.25
    
    # Defino los arrays de parámetros diferentes
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((2,XX.shape[0],XX.shape[1]))
    
    # Primero reviso mis conjuntos de distancias para cada punto del espacio. A esas distancias
    # les calculo el promedio para el subconjunto cuya distancia sea menor que las dist_lim.
    
    for i,dist_crit in enumerate(Criterios):
    
        for fila in range(Dist_JS.shape[0]):
            for columna in range(Dist_JS.shape[1]):
                
                distancias = Dist_JS[fila,columna][Dist_JS[fila,columna] <= dist_crit]
                if distancias.shape[0] == 0:
                    continue
                ZZ[0,fila,columna] = np.mean(distancias)
                ZZ[1,fila,columna] = distancias.shape[0]/Dist_JS.shape[2]
        
        #------------------------------------------------------------------------
        
        # Armo los parámetros del gráfico de promedios
        plt.rcParams.update({'font.size': 44})
        plt.figure("Promedios Distancia JS subconjunto",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        plt.title("Promedios Distancias Jensen-Shannon, distancia corte = {}\n {} vs {}".format(dist_crit,dict_labels[code_y],dict_labels[code_x]))
        
        # Guardo la figura
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistPromSubconj_{}vs{}_r{}.png".format(carpeta,code_y,code_x,i))
        
        plt.pcolormesh(XX,YY,ZZ[0], shading="nearest", cmap = "viridis")
        plt.colorbar()
        
        # Guardo la figura y la cierro
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Promedios Distancia JS subconjunto")
        
        #------------------------------------------------------------------------
        
        # Armo los parámetros del gráfico de promedios
        plt.rcParams.update({'font.size': 44})
        plt.figure("Fraccion simulaciones subconjunto",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        plt.title("Fraccion de simulaciones, distancia corte = {}\n {} vs {}".format(dist_crit,dict_labels[code_y],dict_labels[code_x]))
        
        # Guardo la figura
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/FracSim_{}vs{}_r{}.png".format(carpeta,code_y,code_x,i))
        
        plt.pcolormesh(XX,YY,ZZ[1], shading="nearest", cmap = "bone", vmin = 0, vmax = 1)
        plt.colorbar()
        
        # Guardo la figura y la cierro
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Fraccion simulaciones subconjunto")
    


#-----------------------------------------------------------------------------------------------
    
# Realizo un ajuste de los parámetros Beta y Cos(delta)
    
def Ajuste_DJS(DF_datos,DF_Anes,path,carpeta,Dic_ANES,
               Cd_range,B_range):
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    frac_agente_ind = 1/AGENTES
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_x = Arr_param_x[(Arr_param_x >= Cd_range[0]) & (Arr_param_x <= Cd_range[1])]
    
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    Arr_param_y = Arr_param_y[(Arr_param_y >= B_range[0]) & (Arr_param_y <= B_range[1])]
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    # Armo el vector en el que voy a poner todas las distancias Jensen-Shannon que calcule
    YY_centro = np.reshape(np.array([]),(0,1))
    YY_cruz = np.reshape(np.array([]),(0,1))
    
    XX = np.reshape(np.array([]),(0,5))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo la distribución en hist2d
    
    # Extraigo mis distribuciones sacando el centro, es decir, saco a los agentes que respondieron neutro
    # en ambas encuestas y también extraigo la distribución sacando la cruz, que sería sacar a los agentes que
    # respondieron neutro en alguna encuesta.
    
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    
    # Distribución de encuesta sin el Centro
    
    df_filtered = df_aux[(df_aux[Dic_ANES["code_1"]] != 4) | (df_aux[Dic_ANES["code_2"]] != 4)] # Sólo saca el centro
    hist2d_centro, xedges, yedges, im = plt.hist2d(x=df_filtered[Dic_ANES["code_1"]], y=df_filtered[Dic_ANES["code_2"]], weights=df_filtered[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[Dic_ANES["code_1"]].min()-0.5, df_filtered[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_filtered[Dic_ANES["code_2"]].min()-0.5, df_filtered[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    # Distribución de encuesta sin la Cruz
    
    df_filtered = df_aux[(df_aux[Dic_ANES["code_1"]] != 4) & (df_aux[Dic_ANES["code_2"]] != 4)] # Saca la cruz
    hist2d_cruz, xedges, yedges, im = plt.hist2d(x=df_filtered[Dic_ANES["code_1"]], y=df_filtered[Dic_ANES["code_2"]], weights=df_filtered[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[Dic_ANES["code_1"]].min()-0.5, df_filtered[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_filtered[Dic_ANES["code_2"]].min()-0.5, df_filtered[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    # Armo las distribuciones finalmente
    
    Distr_Enc_Centro = np.reshape(hist2d_centro,(hist2d_centro.shape[0]*hist2d_centro.shape[1],1))
    Distr_Enc_Centro = np.delete(Distr_Enc_Centro,24) # Saco el elemento del centro que tiene un cero
    
    Distr_Enc_Cruz = np.reshape(hist2d_cruz,(hist2d_cruz.shape[0]*hist2d_cruz.shape[1],1))
    Ind_nulos = np.array([3,10,17,21,22,23,24,25,26,27,31,38,45])
    Distr_Enc_Cruz = np.delete(Distr_Enc_Cruz,Ind_nulos)
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        
        #-----------------------------------------------------------------------------------------
        
        Dist_previa_centro = np.zeros(4)
        Dist_previa_cruz = np.zeros(4)
        DistJS_centro = np.zeros((archivos.shape[0],1))
        DistJS_cruz = np.zeros((archivos.shape[0],1))
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Pasos simulados
            # Semilla
            # Matriz de Adyacencia
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales y me armo una distribución en forma de matriz de 7x7
            Opifinales = np.array(Datos[5][:-1], dtype="float")
            Opifinales = Opifinales / EXTRAS
            Distr_Orig = Clasificacion(Opifinales,hist2d_centro.shape[0],hist2d_centro.shape[1],T)
            
            #-----------------------------------------------------------------------------------------
            
            for rotacion in range(4):
                
                # Tomo la distribución original, le doy forma de matriz, la roto y luego la plancho
                Distr_Orig = np.reshape(Distr_Orig, hist2d_centro.shape)
                Distr_Orig = Rotar_matriz(Distr_Orig)
                Distr_Orig = np.reshape(Distr_Orig,(hist2d_centro.shape[0]*hist2d_centro.shape[1],1))
                
                # A partir de la distribución original voy a fabricarme dos distribuciones de simulaciones, una sin el
                # punto central, otra sin la cruz.
                
                # Primero armo la que no tiene el centro removiendo el punto central
                Distr_Sim_centro = np.delete(Distr_Orig,24)
                # Como removí parte de mi distribución, posiblemente ya no esté normalizada
                # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
                # de que otra vez esté normalizada
                if np.sum(Distr_Sim_centro) != 1:
                    agentes_agregar = int((1-np.sum(Distr_Sim_centro))/frac_agente_ind)
                    for i in range(agentes_agregar):
                        ubic_min = np.argmin(Distr_Sim_centro)
                        Distr_Sim_centro[ubic_min] += frac_agente_ind
                # Luego de volver a normalizar mi distribución, si quedaron lugares
                # sin agentes, los relleno
                restar = np.count_nonzero(Distr_Sim_centro == 0)
                ubic = np.argmax(Distr_Sim_centro)
                Distr_Sim_centro[Distr_Sim_centro == 0] = np.ones(restar)*frac_agente_ind
                Distr_Sim_centro[ubic] -= frac_agente_ind*restar
                
                Dist_previa_centro[rotacion] = jensenshannon(Distr_Enc_Centro,Distr_Sim_centro)
                
                
                # Segundo armo la que no tiene la cruz removiendo los puntos en la cruz de la distribución
                Distr_Sim_cruz = np.delete(Distr_Orig, Ind_nulos)
                # Como removí parte de mi distribución, posiblemente ya no esté normalizada
                # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
                # de que otra vez esté normalizada
                if np.sum(Distr_Sim_cruz) != 1:
                    agentes_agregar = int((1-np.sum(Distr_Sim_cruz))/frac_agente_ind)
                    for i in range(agentes_agregar):
                        ubic_min = np.argmin(Distr_Sim_cruz)
                        Distr_Sim_cruz[ubic_min] += frac_agente_ind
                # Luego de volver a normalizar mi distribución, si quedaron lugares
                # sin agentes, los relleno
                restar = np.count_nonzero(Distr_Sim_cruz == 0)
                ubic = np.argmax(Distr_Sim_cruz)
                Distr_Sim_cruz[Distr_Sim_cruz == 0] = np.ones(restar)*frac_agente_ind
                Distr_Sim_cruz[ubic] -= frac_agente_ind*restar
                
                Dist_previa_cruz[rotacion] = jensenshannon(Distr_Enc_Cruz,Distr_Sim_cruz)
                
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            DistJS_centro[repeticion] = np.min(Dist_previa_centro)
            DistJS_cruz[repeticion] = np.min(Dist_previa_cruz)
            
        #------------------------------------------------------------------------------------------
        # Una vez que calculé todas las distancias Jensen-Shannon, concateno eso con mi
        # vector YY, así como concateno mi vector XX
        
        YY_centro = np.concatenate((YY_centro,DistJS_centro))
        YY_cruz = np.concatenate((YY_cruz,DistJS_cruz))
        
        Bcuad = np.ones((len(archivos),1))*PARAM_Y*PARAM_Y
        Blin = np.ones((len(archivos),1))*PARAM_Y
        CDcuad = np.ones((len(archivos),1))*PARAM_X*PARAM_X
        CDlin = np.ones((len(archivos),1))*PARAM_X
        Unos = np.ones((len(archivos),1))
        
        M = np.concatenate((Bcuad,Blin,CDcuad,CDlin,Unos),axis=1)
        XX = np.concatenate((XX,M))
        
    #------------------------------------------------------------------------------------------
    
    # Una vez que tengo armado los vectores XX e YY, ya puedo calcular mis coeficientes

    param_centro = np.matmul(np.linalg.inv(np.matmul(np.transpose(XX),XX)),np.matmul(np.transpose(XX),YY_centro))
    param_cruz = np.matmul(np.linalg.inv(np.matmul(np.transpose(XX),XX)),np.matmul(np.transpose(XX),YY_cruz))
    
    return param_centro,param_cruz

#-----------------------------------------------------------------------------------------------
    
# Ploteo el paraboloide que ajuste en un gráfico 3D

def plot_3d_surface(carpeta, Dic_ANES, func, params_centro, params_cruz, x_range, y_range,
                    SIM_param_x, SIM_param_y,x_samples=100, y_samples=100):
    
    """
    Plot a 3D surface for a given mathematical function.
    
    Parameters:
    - func: The mathematical function to plot. It should take two arguments (x and y) and return a value.
    - x_range: A tuple specifying the range of x values (min, max).
    - y_range: A tuple specifying the range of y values (min, max).
    - x_samples: Number of samples in the x range.
    - y_samples: Number of samples in the y range.
    """
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], x_samples)
    y = np.linspace(y_range[0], y_range[1], y_samples)
    X, Y = np.meshgrid(x, y)
    Z_centro = func(X, Y, params_centro)
    Z_cruz = func(X, Y, params_cruz)

    #--------------------------------------------------------------------------------------
    
    # Armo el gráfico del paraboloide para el caso sin el centro de la distribución
    
    # Create the plot
    direccion_guardado = Path("../../../Imagenes/{}/Paraboloide_ajustado_sin_centro_{}vs{}.png".format(carpeta/"Sin Centro",Dic_ANES["code_2"],Dic_ANES["code_1"]))
    plt.rcParams.update({'font.size': 44})
    fig = plt.figure(figsize=(48,36))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_centro, cmap='viridis')

    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    
    # Format axes ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Adjust tick parameters
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.tick_params(axis='z')
    
    # Labels and title
    ax.set_xlabel(r"${}$".format(SIM_param_x),labelpad = 60)
    ax.set_ylabel(r"${}$".format(SIM_param_y),labelpad = 60)
    ax.set_zlabel('Distancia JS',labelpad = 60)
    ax.set_title('Paraboloide ajustada con distribuciones sin centro \n {} vs {}'.format(Dic_ANES["code_2"],Dic_ANES["code_1"]))

    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close()
    
    #--------------------------------------------------------------------------------------
    
    # Armo el gráfico para el caso sin la cruz central de la distribución
    
    # Create the plot
    direccion_guardado = Path("../../../Imagenes/{}/Paraboloide_ajustado_sin_cruz_{}vs{}.png".format(carpeta/"Sin Cruz",Dic_ANES["code_2"],Dic_ANES["code_1"]))
    plt.rcParams.update({'font.size': 44})
    fig = plt.figure(figsize=(48,36))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_cruz, cmap='viridis')

    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Format axes ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Adjust tick parameters
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.tick_params(axis='z')
    
    # Labels and title
    ax.set_xlabel(r"${}$".format(SIM_param_x),labelpad = 60)
    ax.set_ylabel(r"${}$".format(SIM_param_y),labelpad = 60)
    ax.set_zlabel('Distancia JS',labelpad = 60)
    ax.set_title('Paraboloide ajustada con distribuciones sin cruz \n {} vs {}'.format(Dic_ANES["code_2"],Dic_ANES["code_1"]))

    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close()

#-----------------------------------------------------------------------------------------------
    
# Ploteo el los puntos sobre los que hice el ajuste en un gráfico 3D
    
def plot_3d_scatter(DF_datos,DF_Anes, path, carpeta, Dic_ANES, x_range, y_range,
                    SIM_param_x, SIM_param_y):
    
    """
    Plot a 3D surface for a given mathematical function.
    
    Parameters:
    - func: The mathematical function to plot. It should take two arguments (x and y) and return a value.
    - x_range: A tuple specifying the range of x values (min, max).
    - y_range: A tuple specifying the range of y values (min, max).
    - x_samples: Number of samples in the x range.
    - y_samples: Number of samples in the y range.
    """
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_x = Arr_param_x[Arr_param_x > x_range[0]]
    Arr_param_x = Arr_param_x[Arr_param_x < x_range[1]]
    
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    Arr_param_y = Arr_param_y[Arr_param_y > y_range[0]]
    Arr_param_y = Arr_param_y[Arr_param_y < y_range[1]]
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((XX.shape[0],XX.shape[1],100))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo la distribución en hist2d
    
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[Dic_ANES["code_1"]], y=df_aux[Dic_ANES["code_2"]], weights=df_aux[Dic_ANES["weights"]], vmin=0,
             bins=[np.arange(df_aux[Dic_ANES["code_1"]].min()-0.5, df_aux[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_aux[Dic_ANES["code_2"]].min()-0.5, df_aux[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    Distr_Enc = np.reshape(hist2d,(hist2d.shape[0]*hist2d.shape[1],1))
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        DistJS = np.zeros(archivos.shape[0])
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Pasos simulados
            # Semilla
            # Matriz de Adyacencia
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.array(Datos[5][:-1], dtype="float")
            Opifinales = Opifinales / EXTRAS
            Distr_Sim = np.reshape(Clasificacion(Opifinales,hist2d.shape[0],hist2d.shape[1],T),(hist2d.shape[0]*hist2d.shape[1],1))
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            DistJS[repeticion] = jensenshannon(Distr_Enc,Distr_Sim)
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna,:] = DistJS

    # Create the plot
    direccion_guardado = Path("../../../Imagenes/{}/Scatter de DJS_{}vs{}.png".format(carpeta,Dic_ANES["code_2"],Dic_ANES["code_1"]))
    plt.rcParams.update({'font.size': 44})
    fig = plt.figure(figsize=(40,30))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(100):
        ax.scatter(XX,YY,ZZ[:,:,i], c="blue", marker="o")

    # Labels and title
    ax.set_xlabel(r"${}$".format(SIM_param_x))
    ax.set_ylabel(r"${}$".format(SIM_param_y))
    ax.set_zlabel('Distancia JS')
    ax.set_title('Distancias calculadas \n {} vs {}'.format(Dic_ANES["code_2"],Dic_ANES["code_1"]))

    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close()
    

#-----------------------------------------------------------------------------------------------

# Tomo una matriz y la roto. Repito, roto la matriz como quien gira la cara de un
# cubo Rubik, no estoy rotando el objeto que la matriz representa.

def Rotar_matriz(M):
    
    # Primero miro el tamaño de la matriz que recibí
    n = M.shape[0]
    
    # Armo la matriz P que voy a returnear
    P = np.zeros(M.shape)
    
    # Giro el anillo más externo. Lo hago todo de una.
    for i in range(n):
        P[i,n-1] = M[0,i]
        P[n-1,n-1-i] = M[i,n-1]
        P[n-1-i,0] = M[n-1,n-1-i]
        P[0,i] = M[n-1-i,0]
        
    # Recursivamente mando la parte interna de la matriz M a resolverse
    # con esta misma función.
    if n > 3:
        P[1:n-1,1:n-1] = Rotar_matriz(M[1:n-1,1:n-1])
    elif n == 3:
        P[1:n-1,1:n-1] = M[1:n-1,1:n-1]
    
    return P
