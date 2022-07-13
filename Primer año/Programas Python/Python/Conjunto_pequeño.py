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
    
    for AGENTES in np.unique(Df_archivos_r["n"]):
        
        #---------------------------------------------------------------------------------------------------------------
        
        Conjunto_Mus = np.unique(Df_archivos_r.loc[Df_archivos_r["n"]==AGENTES,"mu"].values)
        
        # Me defino el conjunto de coeficientes de decaimiento mu que voy a recorrer.
        # Desde ahí voy a definir el conjunto de valores Cdelta . Voy a hacer lo mismo que
        # hice con los Cdelta y Alfa, donde definiré un conjunto de Cdelta para cada mu y
        # un conjunto de Alfas para cada Cdelta.
        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        for i_mu,MU in enumerate(Conjunto_Mus):

            Conjunto_Cdelta = np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU),"cdelta"].values)
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
                
                # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
                
                Conjunto_Alfa = np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA),"alfa"].values)
                
                #------------------------------------------------------------------------------------------------------------------------------------------------
                
                for i_alfa,ALFA in enumerate(Conjunto_Alfa):
                    
                    # Graficar me restringe la cantidad de gráficos a armar, cosa de no llenarme de miles de gráficos iguales.
                    Graficar = [1,5]
                    
                    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    for numero,nombre in enumerate(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA), "nombre"]):
                        
                        #-----------------------------------------------------------------------------------------------
                        # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso. En este sistema de Conjunto_pequeño tengo 2 o 3 agentes.
                        
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones,nombre)) 
                        
                        Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                        
                        for i,fila in enumerate(Datos[1:-1:]):
                            Testigos[i] = fila[:-1]
                        
                        # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                        
                        #----------------------------------------------------------------------------------------------------------------------------------
                        
                        # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                        repeticion = int(nombre.strip(".file").split("_")[5].split("=")[1])
                        
                        if repeticion in Graficar:
                        
                            plt.rcParams.update({'font.size': 20})
                            plt.figure("Topico",figsize=(20,15))
                            X = np.arange(Testigos.shape[0])*0.01
                            for sujeto in range(int(AGENTES)):
                                plt.plot(X,Testigos[:,sujeto*2], linewidth = 6)
                                plt.plot(X,Testigos[:,sujeto*2+1], linewidth = 6)
                            plt.xlabel("Tiempo")
                            plt.ylabel("Tópico")
                            plt.grid(alpha = 0.5)
                            plt.annotate(r"$\alpha$={},cos($\delta$)={},$\mu$={},N={}".format(ALFA,CDELTA,MU,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                            plt.savefig("../../Imagenes/{}/TopicosvsT_N={:.0f}_alfa={:.3f}_Cdelta={:.2f}_mu={:.2f}_sim={}.png".format(Carpeta,AGENTES,ALFA,CDELTA,MU,repeticion),bbox_inches = "tight")
                            plt.close("Topico")
                        
                        #-----------------------------------------------------------------------------------------------
                        
        # Acá me voy a tener que hacer el armado de datos para graficar el Conjunto de Hist2D   
         
        Conjunto_Mus = np.unique(Df_archivos_r.loc[Df_archivos_r["n"]==AGENTES,"mu"].values)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        for i_mu,MU in enumerate(Conjunto_Mus):

            Conjunto_Cdelta = np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU),"cdelta"].values)
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
                
                Conjunto_Alfa = np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA),"alfa"].values)
                
                #------------------------------------------------------------------------------------------------------------------------------------------------

                for i_alfa,ALFA in enumerate(Conjunto_Alfa):
                
                    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    plt.rcParams.update({'font.size': 20})
                    fig, ax = plt.subplots(figsize = (20, 15))
                    
                    # Armo el array de Opiniones Finales
                    OpinionesFinales = np.array([])
                    
                    for numero,nombre in enumerate(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA), "nombre"]):
                        
                        #----------------------------------------------------------------------------------------------------------------------------------
                        # Acá levanto los datos de los archivos de Testigos. Estos archivos tienen los siguientes datos:
                        
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones,nombre))
                        
                        # Levanto los datos de la Variación Promedio
                        Opi = np.array([float(x) for x in Datos[len(Datos)-2][:-1]])
                        
                        # Me voy armando un array con las opiniones finales de los agentes a lo largo
                        # de todas las simulaciones
                        
                        OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                        
                        # -------------------------------------------------------------------------------------------------
                    
                    # Acá voy a armar gráficos conjuntos de los Histogramas 2D cosa de que el etiquetado se realice de mejor forma.
                    # Dios quiera que esto salga fácil y rápido.

                    # Ahora grafico las curvas de distribución de ambas opiniones
                    ax.hist2d(OpinionesFinales[0::2],OpinionesFinales[1::2], bins=(50,50), density=True, cmap=plt.cm.Reds)
                    ax.annotate(r"$\alpha$={},cos($\delta$)={},$\mu$={},N={}".format(ALFA,CDELTA,MU,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                            
                    #-------------------------------------------------------------------------------------------
                                        
                    plt.savefig("../../Imagenes/{}/Hist2D_N={:.0f}_alfa={:.2f}_Cdelta={:.2f}_mu={:.2f}.png".format(Carpeta,AGENTES,ALFA,CDELTA,MU),bbox_inches = "tight")
                    plt.close(fig)
                                     
    print("Terminé los gráficos de testigos y los histogramas 2D")
    
    Tiempo()
    
    Df_archivos_r = Df_archivos.loc[Df_archivos["tipo"]=="Varprom"]
    
    # Lo que sigue es lo mismo que más arriba, sólo que para los archivos del tipo Varprom
    
    for AGENTES in np.unique(Df_archivos_r["n"]):
        
        #---------------------------------------------------------------------------------------------------------------
        
        Conjunto_Mus = np.unique(Df_archivos_r.loc[Df_archivos_r["n"]==AGENTES,"mu"].values)

        #------------------------------------------------------------------------------------------------------------------------------------------------
        
        for i_mu,MU in enumerate(Conjunto_Mus):

            Conjunto_Cdelta = np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU),"cdelta"].values)
            
            #------------------------------------------------------------------------------------------------------------------------------------------------
            
            for i_cdelta,CDELTA in enumerate(Conjunto_Cdelta):
                
                Conjunto_Alfa = np.unique(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA),"alfa"].values)
                
                #------------------------------------------------------------------------------------------------------------------------------------------------

                for i_alfa,ALFA in enumerate(Conjunto_Alfa):
                
                    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    
                    # Abro el gráfico para los valores de ALFA y CDELTA correctos
                    
                    Colores2 = cm.rainbow(np.linspace(0,1,Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA), "nombre"].shape[0]))
                    plt.rcParams.update({'font.size': 20})
                    plt.figure("Varprom",figsize = (20, 15))
                
                    for numero,nombre in enumerate(Df_archivos_r.loc[(Df_archivos_r["n"]==AGENTES) & (Df_archivos_r["mu"]==MU) & (Df_archivos_r["cdelta"]==CDELTA) & (Df_archivos_r["alfa"]==ALFA), "nombre"]):
                        
                        # Levanto los datos del archivo original y separo los datos en tres listas.
                        # PARA LEVANTAR DATOS DE LA CARPETA TODA MEZCLADA
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones,nombre))
                        
                        # Levanto los datos de Variación Promedio
                        Var = np.array([float(x) for x in Datos[1][:-1]])
                        
                        # Esto es el tiempo a graficar
                        X = np.arange(len(Var))*0.01
                        
                        # Ahora grafico las curvas de Variación de Opiniones
                        plt.semilogy(X,Var,"--",c = Colores2[numero], linewidth = 4)
                        print(nombre)
                        
                        # ----------------------------------------------------------------------------------------------------------------------------
                    
                    # Acá coloco ciertas marcas importantes para el gráfico
                    
                    CritCorte = 0.0005
                    plt.axhline(CritCorte)
                    plt.annotate(r"$\alpha$={},cos($\delta$)={},$\mu$={},N={}".format(ALFA,CDELTA,MU,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                    plt.grid(alpha = 0.5)
                            
                    #-------------------------------------------------------------------------------------------
                
                    # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
                    # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
                    # en volverse conexo
                    
                    plt.savefig("../../Imagenes/{}/Varprom_N={:.0f}_alfa={:.2f}_Cdelta={:.2f}_mu={:.2f}.png".format(Carpeta,AGENTES,ALFA,CDELTA,MU),bbox_inches = "tight")
                    plt.close("Varprom")
                    
                    #----------------------------------------------------------------------------------------------
                    
    print("Tengo los tiempos de simulación")
    
Tiempo()

