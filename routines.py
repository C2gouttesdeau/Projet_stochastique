##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def position_direction_init(N,l):
#Fonction qui génère les positions et directions initiales des particules
    x_init =[] #liste des positions initiales (abscisses)
    y_init = [] #liste des positions initiales (ordonnées)
    d_init = [] #liste des directions initiales
    for i in range(N):
        x_init.append(np.random.uniform(0,l))
        y_init.append(np.random.uniform(0,l))
        d_init.append(np.random.uniform(0,2*pi))
        
    return x_init, y_init, d_init

def crea_epsilon(eps_x, eps_y):
    return np.array([[eps_x,0],[0,eps_y]])

def distance_rnm(x_n, y_n, x_m, y_m,l):
# Fonction qui calcule la distance entre les individus n et m
    rnm =[]
    X_n = np.transpose([x_n,y_n]) #vecteur position n
    X_m = np.transpose([x_m,y_m]) #vecteur position m
    L = np.transpose([l,l])
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            epsilon = crea_epsilon(i, j)
            rmn = np.append(np.linalg.norm(X_n - (X_m + epsilon*L)))
    return rnm


