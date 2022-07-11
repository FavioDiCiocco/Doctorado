# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:53:51 2021

@author: Favio
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import time
import math
import os

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

#--------------------------------------------------------------------------------

# Voy a definir TODAS las funciones que voy a usar, total definirlas no roba
# tiempo o proceso al programa.


# Esto printea una cantidad de valores cant de un objeto iterable que paso
# en la parte de lista.
def scan(lista,cant=10):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo():
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))


# Esta es la función que uso por excelencia para levantar datos de archivos. Lo
# bueno es que lee archivos de forma general, no necesita que sean csv o cosas así
def ldata(archive):
        f = open(archive)
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data 

    
# Esta función calcula el valor del Alfa crítico que grafico en los mapas de colores
# A diferencia de lo que tenía antes, voy a tener que introducir al GM como una variable
def AlfaC(x,GM):
    T = 2 # Número de tópicos
    K = 3 # Influencia social
    if(x>0):
        alfa = (T-1)/((GM*K)*(T-1+x))
    else:
        alfa = (T-1)/((GM*K)*(T-1-x))
    return alfa


# Voy a definir una función para usar en esta celda, que me permite asignar
# a cada ángulo un color. La idea es que esta función reciba como datos
# el vector a clasificar y la cantidad de pedacitos en la cual divido los
# 360 grados de la circunferencia. Luego me devuelve un número, que va a ser
# el índice en el cual se halla el color de ese vector. El índice lo uso
# para buscar el color en un vector que previamente voy a definir con
# ciertos valores de colores en cada índice. 
# IMPORTANTE: Esto vale sólo para vectores 2D

def Indice_Color(vector,Divisiones):
    # Primero calculo el ángulo
    Vhor = [1,0] # Este vector representa la horizontal
    if np.linalg.norm(vector) != 0 :
        vector_unitario = vector/np.linalg.norm(vector)
        Producto_escalar = np.dot(Vhor,vector_unitario)
        Angulo = np.arccos(Producto_escalar)

        # Le hago ajuste considerando el cuadrante del vector
        if vector[1] < 0:
            Angulo = 2*math.pi-Angulo


        # Ahora calculo el valor de división entera y el Resto
        Delta = (2*math.pi)/Divisiones
        Dividendo = Angulo/Delta
        D = math.floor(Dividendo)
        R = (Dividendo - D) * Delta

        # Compruebo en qué casillero cae el ángulo y returneo el índice
        if R <= Delta/2:
            return D # En este caso el ángulo se encuentra entre (D*Delta-Delta/2,D*Delta+Delta/2]
        elif R > Delta/2:
            return (D+1)%Divisiones # En este caso el ángulo se encuentra entre ((D+1)*Delta-Delta/2,(D+1)*Delta+Delta/2]
    else:
        return 0;


# Voy a definir una función que tome un estado del sistema y a partir de las opiniones
# de los agentes pueda determianr si el sistema se encuentra en un estado de Consenso,
# Polarización Descorrelacionada o Estado ideológico. La idea es que la función reciba
# el estado final del sistema y a partir de este devuelva un string que defina en cual
# de mis tres estados me encuentro. Básicamente la idea de esto es que si mis opiniones
# están muy cerca del cero, entonces estoy en consenso porque ese cae al cero fuerte.
# Entonces pedir que el máximo sea menor a 0,1 es en sí una cota bastante generosa.
# Por el otro lado, el estado ideológico me deja valores sobre una sola diagonal,
# entonces si el producto de la opinión del tópico 1 con la del tópico 2 para
# todos los agentes me da con un único signo, estoy en el caso de estado ideológico.
# Finalmente, si algunos de estos productos me dan positivos y otros negativos,
# entonces estoy en Polarización Descorrelacionada.

def EstadoFinal(Array,Histo,Bins):
    
    Topicos = 2

    # Primero identifico los tres posibles picos
    
    Nbins = len(Bins) # Esto es la cantidad de bins totales en los que dividí mi distribución
    
    Pcero = Histo[int((Nbins-1)/2)] # Esto es cuánto mide el pico en cero
    
    DistNegativa = Histo[0:int((Nbins-1)/2)] # Este es la distribución de las opiniones negativas
    Pmenos = max(DistNegativa) # Esto es la altura del pico de Distribución negativa
    
    DistPositiva = Histo[int((Nbins-1)/2)+1::] # Este es la distribución de las opiniones positivas
    Pmas = max(DistPositiva) # Esto es la altura del pico de Distribución positiva
    
    
    ###########################################################################
    
    # Ahora que tengo los picos, puedo empezar a definir el tema de los estados
    #finales. Arranco por el Consenso
    
    if Pcero == max(Pcero,Pmas,Pmenos):   #Pcero*0.85 > Pmax:  (Esto es la versión con umbral)
        return "Consenso"
    
    ###########################################################################
    
    # Ahora veamos el caso de región de transición. Este estado
    # Lo caracterizo porque el sistema tiene un pico central y picos por fuera
    # que no son definitorios del estado
    
#    Pmediano = min(Pmas,Pmenos)
    
#    if Pmaximo*0.7 < Pcero < Pmaximo*1.3 and Pmaximo*0.7 < Pmas < Pmaximo*1.3 and Pmaximo*0.7 < Pmenos < Pmaximo*1.3 :
#        return "RegionTrans"
    
    ###########################################################################
    
#    indicemenos = np.where(DistNegativa == Pmenos)[0][0]
#    indicemas = np.where(DistPositiva == Pmas)[0][0]
    
    if Pcero == min(Pcero,Pmas,Pmenos):  #and Pcero < Pmediano*0.85: (Esto es de la versión umbral)
        
        # Filtro los agentes que se hayan desviado apenas, cosa
        # de que el criterio final se decida con los agentes que se polarizaron
        # correctamente y no los que terminaron en cualquier lugar cerca del
        # cero.
        ArrayT1 = Array[0::2]
        ArrayT2 = Array[1::2]
        Maximo = max(np.absolute(Array))
        
        OpinionesFiltradas = np.zeros(len(Array))
        
        for agente,x1,x2 in zip(np.arange(len(ArrayT1)),ArrayT1,ArrayT2):
            if abs(x1) > Maximo*0.3 and abs(x2) > Maximo*0.3:
                OpinionesFiltradas[0+agente*Topicos:2+agente*Topicos] = [x1,x2]
        
        # Ahora veamos los otros dos casos. Primero voy a armar
        # un array que tenga las opiniones del tópico 1, y otro
        # con las opiniones del tópico 2.
        
        ArrayCuad = ClasificacionCuadrantes(OpinionesFiltradas)
        
        Cant1 = np.count_nonzero(ArrayCuad == 1)
        Cant2 = np.count_nonzero(ArrayCuad == 2)
        Cant3 = np.count_nonzero(ArrayCuad == 3)
        Cant4 = np.count_nonzero(ArrayCuad == 4)
        
        if Cant2 > 0 and Cant4 > 0 and Cant1 == 0 and Cant3 == 0:
            return "Ideologico"
        elif Cant2 == 0 and Cant4 == 0 and Cant1 > 0 and Cant3 > 0:
            return "Ideologico"
        else:
            return "Polarizacion"
    
    return "RegionTrans"


# Voy a definir una función que tome un array con opiniones del sistema y me 
# diga en qué cuadrante se encuentran cada una de estas coordenadas. Luego a
# la salida me devuelve un array con el cual me voy a armar un histograma

def ClasificacionCuadrantes(Array):
    
    # Primero tomo el array y con sign reviso si sus variables son positivas o negativas.
    # Luego, creo el array Resultado que es lo que voy a returnear.
    # Lo siguiente es crear el array SwitchDic, que va a funcionar como un Switch para los
    # casos que voy a considerar.
    
    Resultado = np.zeros(int(len(Array)/2))
    SwitchDic = dict()
    
    # Defino todos los casos posibles
    
    SwitchDic[(1,1)] = 1
    SwitchDic[(-1,1)] = 2
    SwitchDic[(-1,-1)] = 3
    SwitchDic[(1,-1)] = 4

    
    # Repaso los elementos en Signos para identificar los cuadrantes de mis objetos.
    
    for x1,x2,indice in zip(Array[0::2],Array[1::2],np.arange(len(Array[0::2]))):
        Absolutos = np.abs(np.array([x1,x2]))
        if max(Absolutos)<0.5 or x1==0 or x2==0:
            Resultado[indice] = 0
        else:
            Signos = np.sign(np.array([x1,x2]))
            Resultado[indice] = SwitchDic[(Signos[0],Signos[1])]
  
    return Resultado


# Acá lo que voy a hacer es preparar los colores que voy a usar para definir los puntos finales
# de las trayectorias de las opiniones

Divisiones = 144
color=cm.rainbow(np.linspace(0,1,Divisiones))


# Lo que hice acá es definir una ¿lista? que tiene en cada casillero los datos que definen un color.
# Tiene diferenciados 144 colores, es decir que tengo un color para cada región de 2.5 grados. Estas regiones
# las voy a distribuir centrándolas en cada ángulo que cada color representa. Por lo tanto,
# Los vectores que tengan ángulo entre -1.25º y 1.25º tienen el primer color. Los que tengan entre
# 1.25º y 3.75º tienen el segundo color. Y así. Por tanto yo tengo que hallar una fórmula que para
# cada ángulo le asigne el casillero que le corresponde en el vector de color. Luego, cuando grafique
# el punto, para el color le agrego un input que sea: c = color[n]



# Estas son todas las funciones que voy a necesitar. Lo último no es una función,
# es la lista de colores que necesitaba para usar la función Indice_Color.
# Ahora voy a definir la parte que se encarga de levantar los nombre y el path
# de los archivos.

t0 = time.time()

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

T=2 # Defino acá el número de tópicos porque es algo que no cambia por ahora,
# pero no tenía dónde más definirlo

SuperDiccionario = dict()

for Carpeta in ["Conjunto_pequeño"]:

   # CÓDIGO PARA LEVANTAR ARCHIVOS DE UNA CARPETA CON TODOS LOS ARCHIVOS MEZCLADOS
    
    CarpCheck=[[root,files] for root,dirs,files in os.walk("../{}".format(Carpeta))]
    
    # El elemento en la posición x[0] es el nombre de la carpeta
    
    for x in CarpCheck:
        # dada = x[0].split("\\")
        Archivos_Datos = [nombre for nombre in x[1]]
        Archivos_Datos.insert(0,x[0])
    
    #-------------------------------------------------------------------------------------------------------
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Varprom_alfa=$_N=$_Cosd=$_mu=$_Iter=$.file"
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
    
    Df_archivos_r = Df_archivos.loc[Df_archivos["tipo"]=="Testigos"]
    
    for AGENTES in np.unique(Df_archivos["n"]):
        
        #---------------------------------------------------------------------------------------------------------------
        
        Conjunto_Mus = np.sort(np.unique(Df_archivos_r.loc[Df_archivos_r["n"]==AGENTES,"mu"].values))
        
        # Me defino el conjunto de coeficientes de decaimiento mu que voy a recorrer.
        # Desde ahí voy a definir el conjunto de valores Cdelta . Voy a hacer lo mismo que
        # hice con los Cdelta y Alfa, donde definiré un conjunto de Cdelta para cada mu y
        # un conjunto de Alfas para cada Cdelta.
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        for i_mu,MU in enumerate(Conjunto_Mus):

            Conjunto_Cdelta = np.sort(np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU),"cdelta"].values))
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
                
                # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
                
                Conjunto_Alfa = np.sort(np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA),"alfa"].values))
                
                #------------------------------------------------------------------------------------------------------------------------------------------------
                
                for i_alfa,ALFA in enumerate(Conjunto_Alfa):
                    
                    # Graficar me restringe la cantidad de gráficos a armar, cosa de no llenarme de miles de gráficos iguales.
                    Graficar = [1,5]
                    
                    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    for numero,nombre in enumerate(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA), "nombre"]):
                        
                        #-----------------------------------------------------------------------------------------------
                        # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso. En este sistema de Conjunto_pequeño tengo 2 o 3 agentes.
                        
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones,nombre)) 
                        
                        Agentes = [ agent for agent in range(int(len(Datos[1])/2))] # Lo hago un range porque al final no importa cuál testigo es cual, es pura casualidad.
                        
                        Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                        
                        
                        for i,fila in enumerate(Datos[1::]):
                            Testigos[i] = fila[:-1]
                            
                        # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                        
                        #----------------------------------------------------------------------------------------------------------------------------------
                        
                        # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                        repeticion = int(nombre.strip(".file").split("_")[5].split("=")[1])
                        
                        if repeticion in Graficar and ALFA in [0,2] and CDELTA in [0,1]:
                        
                            plt.rcParams.update({'font.size': 18})
                            plt.figure("Topico",figsize=(12,8))
                            X = np.arange(Testigos.shape[0])*0.01
                            for sujeto in range(int(len(Testigos[0])/2)):
                                plt.plot(X,Testigos[:,sujeto*2], linewidth = 6)
                                plt.plot(X,Testigos[:,sujeto*2+1], linewidth = 6)
                            plt.xlabel("Tiempo")
                            plt.ylabel("Tópico")
                            plt.grid(alpha = 0.5)
                            plt.annotate(r"$\alpha$={},cos($\delta$)={},$\mu$={},N={}".format(ALFA,CDELTA,MU,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                            plt.savefig("../../Imagenes/{}/TopicosvsT_alfa={:.3f}_Cdelta={:.2f}_mu={:.2f}_sim={}.png".format(Carpeta,ALFA,CDELTA,MU,repeticion),bbox_inches = "tight")
                            plt.close("Topico")
                        
                        #-----------------------------------------------------------------------------------------------
    print("Terminé los gráficos de testigos")
    
    Df_archivos_r = Df_archivos.loc[Df_archivos["tipo"]=="Varprom"]
    
    # Lo que sigue es lo mismo que más arriba, sólo que para los archivos del tipo Varprom
    
    for AGENTES in np.unique(Df_archivos["n"]):
        
        #---------------------------------------------------------------------------------------------------------------
        
        Conjunto_Mus = np.sort(np.unique(Df_archivos_r.loc[Df_archivos_r["n"]==AGENTES,"mu"].values))

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        for i_mu,MU in enumerate(Conjunto_Mus):

            Conjunto_Cdelta = np.sort(np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU),"cdelta"].values))
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
                
                Conjunto_Alfa = np.sort(np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA),"alfa"].values))
                
                #------------------------------------------------------------------------------------------------------------------------------------------------

                for i_alfa,ALFA in enumerate(Conjunto_Alfa):
                
                    Graficar = [1,5]
                    
                    # Inicio mis arrays para guardar los datos de Tiempos de Simulación y de Promedios
                    
                    TideSi = np.zeros(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA)].shape[0])
                    Promedios = np.zeros(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA)].shape[0])
                    
                    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    for numero,nombre in enumerate(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA), "nombre"]):
                        
                        # Levanto los datos del archivo original y separo los datos en tres listas.
                        # PARA LEVANTAR DATOS DE LA CARPETA TODA MEZCLADA
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones,nombre))
                        
                        # Levanto los datos de Variación Promedio
                        Var = np.array([float(x) for x in Datos[1][:-1]])
                        
                        #-----------------------------------------------------------------------------------------------

                        # Voy a ir armando mi array con los valores de Tiempo de Simulación.
                        # Lo voy a llamar TideSi.
                        
                        TideSi[numero] = Var.shape[0]*0.01 # 0,01 es el paso temporal dt
                        
                        #--------------------------------------------------------------------------------------------------
                            
                    print("Tengo las variaciones promedio".format(MU,CDELTA))
                    Tiempo()
                    
                    #----------------------------------------------------------------------------------------------
                    
            
        #---------------------------------------------------------------------------------------------------------
                    
        # Acá me voy a tener que hacer el armado de datos para graficar el Conjunto de Variaciones Promedio 
        
        TIPO = "Opiniones"
        
        for i_mu,MU in enumerate(Conjunto_Mus):
         
        # Ahora armo el Conjunto C_delta para recorrer mi espacio
            Conjunto_Cdelta = list(SuperDiccionario[Carpeta][TIPO][AGENTES][MU].keys())
            Conjunto_Cdelta.sort()
            
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
            
                # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
                # De nuevo, lo ordeno porque el SuperDiccionario no los tiene necesariamente ordenados.
                
                Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][MU][CDELTA].keys())
                Conjunto_Alfa.sort()
                
                # Acá estoy haciendo los gráficos de Varprom individuales. Si quisiera los gráficos de manera conjunta debería revisar la parte
                # que está comentada más abajo.
                
                for i_alfa,ALFA in enumerate(Conjunto_Alfa):
                    
                    Colores2 = cm.rainbow(np.linspace(0,1,len(SuperDiccionario[Carpeta][TIPO][AGENTES][MU][CDELTA][ALFA])))
                    plt.rcParams.update({'font.size': 24})
                    fig, ax = plt.subplots(figsize = (20, 15))
                    
                    for numero,nombre in enumerate(SuperDiccionario[Carpeta][TIPO][AGENTES][MU][CDELTA][ALFA]):
                        
                        #----------------------------------------------------------------------------------------------------------------------------------
                        # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                        # Opinión Inicial del sistema
                        # Variación Promedio
                        # Opinión Final
                        # Semilla
                        
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                        
                        # Levanto los datos de la Variación Promedio
                        Varprom = np.array([float(x) for x in Datos[3][:-1]])
                        
                        # ----------------------------------------------------------------------------------------------------------------------------
                        
                        # Esto es el tiempo a graficar
                        X = np.arange(len(Varprom))*0.01
                        
                        # Ahora grafico las curvas de Variación de Opiniones
                        ax.semilogy(X,Varprom,"--",c = Colores2[numero], linewidth = 4)
                        
                        # ----------------------------------------------------------------------------------------------------------------------------
                    
                    # Acá coloco ciertas marcas importantes para el gráfico
                    
                    CritCorte = 0.0005
                    ax.axhline(CritCorte)
                    ax.annotate(r"$\alpha$={},cos($\delta$)={},$\mu$={},N={}".format(ALFA,CDELTA,MU,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                    ax.grid(alpha = 0.5)
                            
                #-------------------------------------------------------------------------------------------
                
                    # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
                    # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
                    # en volverse conexo
                    
                    plt.savefig("../../Imagenes/{}/Varprom_alfa={:.2f}_Cdelta={:.2f}_mu={:.2f}.png".format(Carpeta,ALFA,CDELTA,MU),bbox_inches = "tight")
                    plt.close(fig)
                
                """
                # Inicio la figura con los múltiples gráficos de Variación Promeido
                plt.rcParams.update({'font.size': 24})
                fig = plt.figure("Varprom",figsize=(64,36))
                gs = fig.add_gridspec(6,5,hspace=0,wspace=0)
                axs = gs.subplots(sharex=True, sharey=True)
                
                # Estas son las columnas y filas a recorrer para el gráfico
                Columnas = [0,1,2,3,4]*6
                Filas = [i for i in range(0,6) for j in range(0,5)]
                
                for ALFA,fila,columna in zip(Conjunto_Alfa[0:30],Filas,Columnas):
                    
                    # Defino los colores para graficar las variaciones Promedio
                    Colores2 = cm.rainbow(np.linspace(0,1,len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA])))
                    
                    for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                    
                        #----------------------------------------------------------------------------------------------------------------------------------
                        # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                        # Opinión Inicial del sistema
                        # Variación Promedio
                        # Opinión Final
                        # Semilla
                        
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                        
                        # Levanto los datos de la Variación Promedio
                        Varprom = np.array([float(x) for x in Datos[3][1::]])
                        
                    # ----------------------------------------------------------------------------------------------------------------------------
                        
                        # Esto es el tiempo a graficar
                        X = np.arange(len(Varprom))*0.01
                        
                        # Ahora grafico las curvas de Variación de Opiniones
                        axs[fila,columna].semilogy(X,Varprom,"--",c = Colores2[numero],linewidth = 4)
                        
                    # ----------------------------------------------------------------------------------------------------------------------------
                    
                    # Acá coloco ciertas marcas importantes para el gráfico
                    
                    CritCorte = 0.0005
                    axs[fila,columna].axhline(CritCorte)
                    axs[fila,columna].annotate(r"$\alpha=${}".format(ALFA), xy=(0.8,0.85),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                    axs[fila,columna].grid()
                            
                #-------------------------------------------------------------------------------------------
                
                # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
                # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
                # en volverse conexo
                
                plt.savefig("../Imagenes/RedAct/{}/Varprom_Cdelta={:.2f}.png".format(Carpeta,CDELTA),bbox_inches = "tight")
                plt.close("Varprom")
                    
                """
            #-----------------------------------------------------------------------------------------------------------
                    
        # Acá me voy a tener que hacer el armado de datos para graficar el Conjunto de Hist2D   
        
        for i_mu,MU in enumerate(Conjunto_Mus):
         
        # Ahora armo el Conjunto C_delta para recorrer mi espacio
            Conjunto_Cdelta = list(SuperDiccionario[Carpeta][TIPO][AGENTES][MU].keys())
            Conjunto_Cdelta.sort()
        
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
            
                # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
                # De nuevo, lo ordeno porque el SuperDiccionario no los tiene necesariamente ordenados.
                
                Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][MU][CDELTA].keys())
                Conjunto_Alfa.sort()
                
                # Acá estoy haciendo los gráficos de Hist2D individuales. Si quisiera los gráficos de manera conjunta debería revisar la parte
                # que está comentada más abajo.
                
                for ialfa,ALFA in enumerate(Conjunto_Alfa):
                    
                    plt.rcParams.update({'font.size': 24})
                    fig, ax = plt.subplots(figsize = (20, 15))
                    
                    # Armo el array de Opiniones Finales
                    OpinionesFinales = np.array([])
                    
                    for numero,nombre in enumerate(SuperDiccionario[Carpeta][TIPO][AGENTES][MU][CDELTA][ALFA]):
                        
                        #----------------------------------------------------------------------------------------------------------------------------------
                        # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                        # Opinión Inicial del sistema
                        # Variación Promedio
                        # Opinión Final
                        # Semilla
                        
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                        
                        # Levanto los datos de la Variación Promedio
                        Opi = np.array([float(x) for x in Datos[5][:-1]])
                        
                        # Me voy armando un array con las opiniones finales de los agentes a lo largo
                        # de todas las simulaciones
                        
                        OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                        
                        # -------------------------------------------------------------------------------------------------
                    
                    # Acá voy a armar gráficos conjuntos de los Histogramas 2D cosa de que el etiquetado se realice de mejor forma.
                    # Dios quiera que esto salga fácil y rápido.
                    
                    # OpiMaxima = max(np.absolute(OpinionesFinales))
                    # OpiExtremos = np.array([OpiMaxima,OpiMaxima])
                    # Opiniones2D = np.concatenate((OpinionesFinales,OpiExtremos), axis=None)
                    

                    # Ahora grafico las curvas de distribución de ambas opiniones
                    # ax.hist2d(Opiniones2D[0::2],Opiniones2D[1::2], bins=(79,79), density=True, cmap=plt.cm.Reds)
                    ax.hist2d(OpinionesFinales[0::2],OpinionesFinales[1::2], bins=(79,79), density=True, cmap=plt.cm.Reds)
                    # if ALFA == 2 and CDELTA >= 0:
                        # ax.set_xlim(0,150)
                        # ax.set_ylim(0,150)
                    ax.annotate(r"$\alpha$={},cos($\delta$)={},$\mu$)={},N={}".format(ALFA,CDELTA,MU,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                            
                    #-------------------------------------------------------------------------------------------
                    
                    # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
                    # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
                    # en volverse conexo
                    
                    plt.savefig("../../Imagenes/{}/Hist2D_alfa={:.2f}_Cdelta={:.2f}_mu={:.2f}.png".format(Carpeta,ALFA,CDELTA,MU),bbox_inches = "tight")
                    plt.close(fig)
        
        
            
            """
            
            # Inicio la figura con los múltiples gráficos de Variación Promeido
            plt.rcParams.update({'font.size': 24})
            fig = plt.figure("ConjuntoHist2D",figsize=(64,36))
            gs = fig.add_gridspec(6,5,hspace=0,wspace=0)
            axs = gs.subplots(sharex=True, sharey=True)
            
            # Estas son las columnas y filas a recorrer para el gráfico
            Columnas = [0,1,2,3,4]*6
            Filas = [i for i in range(0,6) for j in range(0,5)]
            
            # Reinicio la OpiMaxima a 0
            OpiMaxima = 0
            
            for ALFA in Conjunto_Alfa[0:30]:
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                        
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                
                    #-----------------------------------------------------------------------------------------------
                        
                    # Acá voy armando el cálculo de la OpiMáxima
                    
                    OpiMaxima = max(max(Opi),OpiMaxima)
                    
            
            for ALFA,fila,columna in zip(Conjunto_Alfa[0:30],Filas,Columnas):
                
                # Armo el array de Opiniones Finales
                OpinionesFinales = np.array([])
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                        
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                    
                    # Me voy armando un array con las opiniones finales de los agentes a lo largo
                    # de todas las simulaciones
                    
                    OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                    
                        
                # -------------------------------------------------------------------------------------------------
                
                # Acá voy a armar gráficos conjuntos de los Histogramas 2D cosa de que el etiquetado se realice de mejor forma.
                # Dios quiera que esto salga fácil y rápido.
                
                OpiMaxima = max(np.absolute(OpinionesFinales))
                OpiExtremos = np.array([OpiMaxima,OpiMaxima,-OpiMaxima,OpiMaxima,OpiMaxima,-OpiMaxima,-OpiMaxima,-OpiMaxima])
                Opiniones2D = np.concatenate((OpinionesFinales,OpiExtremos), axis=None)
                

                # Ahora grafico las curvas de distribución de ambas opiniones
                axs[fila,columna].hist2d(Opiniones2D[0::2],Opiniones2D[1::2], bins=(79,79), density=True, cmap=plt.cm.Reds)
                axs[fila,columna].annotate(r"$\alpha=${}".format(ALFA), xy=(0.8,0.85),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                        
            #-------------------------------------------------------------------------------------------
            
            # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
            # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
            # en volverse conexo
            
            plt.savefig("../Imagenes/RedAct/{}/Conjunto_Hist2D_Cdelta={:.2f}.png".format(Carpeta,CDELTA),bbox_inches = "tight")
            plt.close("ConjuntoHist2D")
        
        #-----------------------------------------------------------------------------------------------------------

        
        
        # Acá me armo los gráficos de Conjunto de Distribución de Opinones

        for CDELTA,icdelta in zip(Conjunto_Cdelta,np.arange(len(Conjunto_Cdelta))):
        
            # Defino los alfas que voy a usar en el gráfico de Distribución de Opiniones
            
            plt.rcParams.update({'font.size': 24})
            
            fig = plt.figure("Distribucion Opiniones",figsize=(64,36))
            
            gs = fig.add_gridspec(4,4, hspace = 0)
            axs = gs.subplots(sharex=True)
            
            # Alfas_Dist = [Conjunto_Alfa[int(indice*math.floor(len(Conjunto_Alfa)/16))] for indice in range(0,16)]
            
            Columnas = [0,1,2,3]*4
            Filas = [i for i in range(0,4) for j in range(0,4)]

            Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA].keys())
            Conjunto_Alfa.sort()
            
            
            # Reinicio la OpiMaxima a 0
            OpiMaxima = 0
            
            for ALFA in Conjunto_Alfa[0:30]:
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                        
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                
                    #-----------------------------------------------------------------------------------------------
                        
                    # Acá voy armando el cálculo de la OpiMáxima
                    
                    OpiMaxima = max(max(Opi),OpiMaxima)
                    
                    #-----------------------------------------------------------------------------------------------
            
            
            for ALFA,fila,columna in zip(Conjunto_Alfa[0:16],Filas,Columnas):
                
                OpinionesFinales = np.array([])

                #-------------------------------------------------------------------------------------
                for nombre in SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]:
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                    
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                    
                    # Me voy armando un array con las opiniones finales de los agentes a lo largo
                    # de todas las simulaciones
                    
                    OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                    
                    
                        
            # -------------------------------------------------------------------------------------------------
                
                # Acá voy a armar los gráficos de las proyecciones de las opiniones de los agentes. Para eso simplemente
                # voy a tomar los valores de opiniones de los agentes de una simulación, calcularle el histograma
                # con np.histogram y luego graficar eso como líneas.
                
                OpiMaxima = max(np.absolute(OpinionesFinales))
                # if OpiMaxima < 150:
                #     Bordes = np.round(np.linspace(-OpiMaxima,OpiMaxima,22),5)
                # else:
                #     BordeI = np.mean(OpinionesFinales[OpinionesFinales < 0])-30
                #     BordeD = np.mean(OpinionesFinales[OpinionesFinales > 0])+30
                #     BordeMax = max(abs(BordeI),BordeD)
                #     Bordes = np.round(np.linspace(-BordeMax,BordeMax,int(OpiMaxima/50)*2),3)
                # else:
                    # Bordes = np.round(np.linspace(-60,60,40),3)
                Bordes = np.round(np.linspace(-60,60,40),3)
                Bins = (Bordes[1::]+Bordes[0:len(Bordes)-1])/2
                
                Histo,nada = np.histogram(OpinionesFinales,bins=Bordes)
                
                ResultadoEF = EstadoFinal(OpinionesFinales,Histo,Bins)
                # if ALFA == 0:
                #     print("Los valores del histograma de alfa=0 son:")
                #     print(Histo)
                
                # Armo los histogramas correspondientes
                Histo,nada = np.histogram(OpinionesFinales,bins=Bordes,density=True)
                
                # Defino el tema de los colores
                if ResultadoEF == "Consenso":
                    color = "green"
                elif ResultadoEF == "Polarizacion":
                    color = "blue"
                elif ResultadoEF == "Ideologico":
                    color = "red"
                elif ResultadoEF == "RegionTrans":
                    color = "purple"
                
                
                # Ahora grafico las curvas de distribución de ambas opiniones
                axs[fila,columna].plot(Bins,Histo,"-o",color = color,linewidth = 4,markersize = 8, label="Alfa = {}".format(ALFA))
                axs[fila,columna].legend()
                axs[fila,columna].grid()
            
            plt.savefig("../Imagenes/RedAct/{}/Distribucion_opiniones_Cdelta={:.2f}.png".format(Carpeta,CDELTA),bbox_inches = "tight")
#                plt.show()
            plt.close("Distribucion Opiniones")
        
        #--------------------------------------------------------------------------------------------------------
        
        
        # Estos son los parámetros que definen el tamaño del gráfico, tamaño de la letra y nombres de
        # los ejes. Luego de eso guardo la figura y la cierro. Esto es para la figura de
        # TdO.
        
        # Este gráfico me coloca los puntos finales de las opiniones con colores y me 
        # pone un cartelito marcando cuál es el estado final del sistema. Voy a agregar
        # que se dibuje un círculo marcando la región donde se encuentran la mayoría de
        # las opiniones incialmente, puntos negros para las opiniones iniciales y agregar
        # el hecho de que los límites de los gráficos tienen que ser entre -30 y 30
        
        # Armo los valores X e Y que me dibujan el círculo
        Tita = np.linspace(0,2*math.pi,1000)
        Xcirc = 2*math.sqrt(2.5)*np.cos(Tita)
        Ycirc = 2*math.sqrt(2.5)*np.sin(Tita)
        
        
        plt.rcParams.update({'font.size': 18})
        plt.figure("Grafico Opiniones",figsize=(20,15))
        
        # Grafico los puntos iniciales de las opiniones
        for x1,x2 in zip (OpinionesIniciales[0::2],OpinionesIniciales[1::2]):
            plt.plot(x1,x2, "o" ,c = "black", markersize=5, alpha = 0.5)
        
        # Grafico los puntos finales de las opiniones
        for x1,x2 in zip (OpinionesFinales[0::2],OpinionesFinales[1::2]):
            indice = Indice_Color(np.array([x1,x2]),Divisiones)
            plt.plot(x1,x2, "o" ,c = color[indice], markersize=10)
        
        # Grafico el círculo en distancia 2 sigma del cero
        plt.plot(Xcirc,Ycirc,"--",linewidth=5,alpha=0.6)
        
        #            plt.tick_params(left=False,
        #                bottom=False,
        #                labelleft=False,
        #                labelbottom=False)
        plt.xlabel(r"$x^1$")
        plt.ylabel(r"$x^2$")
#                    #            plt.title(r"Trayectoria de las opiniones en el espacio de tópicos para $\alpha$={},cos($\delta$)={} y N={}".format(ALFA,CDELTA,AGENTES))
##                    plt.xlim((xmin,xmax))
##                    plt.ylim((ymin,ymax))
        plt.annotate(r"$\alpha$={},cos($\delta$)={},N={}".format(ALFA,CDELTA,AGENTES), xy=(0.7,0.85),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
        plt.savefig("../Imagenes/RedAct/{}/Grafico_opiniones_alfa={:.3f}_Cdelta={:.2f}_N={}.png".format(Carpeta,ALFA,CDELTA,AGENTES),bbox_inches = "tight")
        plt.close("Grafico Opiniones")
        """

        #------------------------------------------------------------------------------------------------
                
                        
Tiempo()
