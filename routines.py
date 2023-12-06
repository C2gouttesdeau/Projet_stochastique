##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from cmath import *
import time
import itertools

def position_direction_init(N,l):
#Fonction qui génère les positions et directions initiales des particules
###
# Entrées :
###
# N : nombre de particules , l : longueur de la boite
###
# Sortie : x_init, y_init, theta_init : positions et directions initiales vecteurs de taille N
###
    x_init = np.random.uniform(0, l, N)
    y_init = np.random.uniform(0, l, N)
    theta_init = np.random.uniform(0, 2*np.pi, N)
    return x_init, y_init, theta_init
    
def crea_epsilon(eps_x, eps_y):
    return np.diag([eps_x, eps_y])

def distance_rnm(x_n, y_n, x_m, y_m,l):
# Fonction qui calcule la distance entre les individus n et m
# x_n, y_n : coordonnées de l'individu n
# x_m, y_m : coordonnées de l'individu m
# l : longueur de la boite
    rnm =[]
    X_n = np.transpose([x_n,y_n]) #vecteur position n
    X_m = np.transpose([x_m,y_m]) #vecteur position m
    L = np.transpose([l,l])
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            epsilon = crea_epsilon(i, j)
            rnm.append(np.linalg.norm(X_n - (X_m + epsilon@L)))
    return min(rnm)

def unit_vector(theta):
# Fonction qui calcule le vecteur unitaire associé à un angle theta
    x = np.cos(theta)
    y = np.sin(theta)
    return x,y

def update_position_direction(N,l,a,v0,dt,eta,x_t,y_t,theta_t):
# Fonction qui calcule les positions et directions des individus à l'instant
###
# Entrées :
###
# N,l,a,v0,dt,eta: nombre d'individus, longueur de la boite, échelle de longueur, vitesse, pas de temps, paramètre de bruit
# x_t, y_t, theta_t : positions à l'instant t et directions à l'instant t vecteurs de taille N
###
# Sortie : x_tfut, y_tfut, theta_tfut : positions et directions à l'instant t+dt vecteurs de taille N
###

    # Initialisation des vecteurs
    x_tfut=np.zeros((N,1))
    y_tfut=np.zeros((N,1))
    theta_tfut=np.zeros((N,1))
    
    for i in range(N):
        #Calcul des directions à l'instant t+dt pour la particule i
        time_init = time.time()
        arg = 0 + 0*1j
        for j in range(N): 
            #on parcourt les particules y compris la i car on ajoute une phase 
            #et si pas de particule on garde la même direction + var aléatoire
            rnm = distance_rnm(x_t[i], y_t[i], x_t[j], y_t[j],l)
            if rnm <= a :   
                arg = arg + (cos(theta_t[j]) + sin(theta_t[j])*1j)
        theta_tfut[i] = phase(arg) + eta*var_al_unif()
        time_end = time.time()
        print(f"Le temps de calcul de la direction de la particule {i} est {round(time_end-time_init,5)} secondes.")  
        #Calcul des positions à l'instant t+dt pour la particule i
        time_init = time.time()
        x_tfut[i] = (x_t[i] + v0*dt*unit_vector(theta_tfut[i])[0]) % l
        y_tfut[i] = (y_t[i] + v0*dt*unit_vector(theta_tfut[i])[1]) % l  
        time_end = time.time()
        print(f"Le temps de calcul de la position de la particule {i} est {round(time_end-time_init,5)} secondes.")
    return np.array(x_tfut), np.array(y_tfut),np.array(theta_tfut)

def Solveur(N,l,a,v0,dt,eta,Nt):
# Fonction qui calcule les positions et directions des individus à chaque instant
###
# Entrées :
###
# N,l,a: nombre d'individus, longueur de la boite, échelle de longueur
# v0,dt,eta,Nt :vitesse, pas de temps, paramètre de bruit, nombre d'itérations
###
# Sortie : x_sol, y_sol, theta_sol : positions et directions à chaque instant matrice de taille Nt*N

    x_t, y_t, theta_t = position_direction_init(N,l) #initialisation
    x_sol = x_t.reshape(1, -1)
    y_sol = y_t.reshape(1, -1)
    theta_sol = theta_t.reshape(1, -1)

    for i in range(Nt-1):
        time_init = time.time() 
        x_t, y_t, theta_t = update_position_direction(N,l,a,v0,dt,eta,x_t,y_t,theta_t)
        # Ajout des vecteurs aux matrices
        x_sol = np.vstack((x_sol, x_t.reshape(1, -1)))
        y_sol = np.vstack((y_sol, y_t.reshape(1, -1)))
        theta_sol = np.vstack((theta_sol, theta_t.reshape(1, -1)))
        time_end = time.time()
        print(f"Le temps de calcul de la solution à l'itération {i} est {round(time_end-time_init,5)} secondes.")
    return x_sol , y_sol, theta_sol

def var_al_unif():
    return random.uniform(-pi,pi)