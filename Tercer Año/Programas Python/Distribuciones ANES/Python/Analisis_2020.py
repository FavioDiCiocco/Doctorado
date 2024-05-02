#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:37:42 2024

@author: favio
"""

#############################################################################################
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import time
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
from scipy.spatial.distance import jensenshannon

import warnings
warnings.filterwarnings('ignore')
#############################################################################################

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
        
#--------------------------------------------------------------------------------
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo(t0):
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))

t0 = time.time()

#############################################################################################

# Cargo el archivo de datos total

filename = "../ANES_2020/anes_timeseries_2020.dta"
df_raw_data = pd.read_stata(filename)

#############################################################################################

# Brief description of the codes
dict_labels = {'V201200':'Liberal-Conservative self Placement', 'V201225x':'Voting as duty or choice','V201231x':'Party Identity',
               'V201246':'Spending & Services', 'V201249':'Defense Spending', 'V201252':'Gov-private Medical Insurance',
               'V201255':'Guaranteed job Income', 'V201258':'Gov Assistance to Blacks', 'V201262':'Environment-Business Tradeoff',
               'V201342x':'Abortion Rights Supreme Court', 'V201345x':'Death Penalty','V201356x':'Vote by mail',
               'V201362x':'Allowing Felons to vote', 'V201372x':'Helpful-Harmful if Pres didnt have to worry about Congress',
               'V201375x':'Restricting Journalist access', 'V201382x':'Corruption increased or decreased since Trump',
               'V201386x':'House impeachment decision', 'V201405x':'Require employers to offer paid leave to parents',
               'V201408x':'Allow to refuse service to same sex couples', 'V201411x':'Transgender Policy', 'V201420x':'Birthright Citizenship',
               'V201423x':'Should children brought illegally be sent back','V201426x':'Wall on border with Mexico',
               'V201429':'Best way to deal with Urban Unrest'}

labels = ['V201200','V201225x','V201231x','V201246','V201249','V201252','V201255','V201258','V201262','V201342x','V201345x',
          'V201356x','V201362x','V201372x','V201375x','V201382x','V201386x','V201405x','V201408x','V201411x','V201420x',
          'V201423x','V201426x','V201429']

#############################################################################################

df_data_aux = df_raw_data[labels]
df_data = pd.DataFrame()

for code in labels:
    df_data[code] = df_data_aux[code].apply(data_processor)
    
df_data[['V200010a','V200010b']] = df_raw_data[['V200010a','V200010b']]

#############################################################################################

# Gráfico de dos preguntas simultáneas con distribuciones individuales en los ejes


weights = 'V200010a'
for i,code_1 in enumerate(labels):
    for code_2 in labels[i+1::]:
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        plt.rcParams.update({'font.size': 28})
        # plt.figure(figsize=(40,21))
        sns.jointplot(df_aux, x=code_1, y=code_2, kind="hist", vmin=0, cmap='inferno', height = 15,
                      joint_kws={'discrete': True, 'weights': df_aux[weights]}, 
                      marginal_kws={'discrete': True, 'weights': df_aux[weights]})
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
        plt.gca().invert_yaxis()
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/Datos/{}vs{}.png".format(code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()


Tiempo(t0)