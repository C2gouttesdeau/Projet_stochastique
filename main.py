####           Projet Stoch           #####
###########################################

##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

##### Import files #####
from routines import *


l=10 #longueur de la boite

N=1000 #nombre de particules

##### Initialisation #####

#Initialisation des positions et directions
x_init, y_init, d_init = position_direction_init(N,l)

#Affichage des positions initiales
plt.figure()
plt.scatter(x_init,y_init)
plt.xlim(0,l)
plt.ylim(0,l)
plt.show()

print(distance_rnm(0, 0, l/2, l/2,l))



# def Theta_n_tdt(Theta_n_t):
    
#     return 


