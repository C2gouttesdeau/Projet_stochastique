##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cmath import *
from scipy.spatial import distance
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.spatial.distance import cdist

from routines import *  

def distance_indice_matrix(x_t,y_t,l,N,a):
#Fonction qui renvoie une matrice de 0 et de 1
# 1 si les particule (i,j) de la matrices sont à une distance inférieur à a
# 0 sinon 
    indice_dist = np.eye(N) #les particules sont à une ditance nul d'elles même 
    L = np.array([l, l])
    for i in range(N):
        X_i= np.array([x_t[i], y_t[i]])
        for j in range(i+1,N):
            X_j = np.array([x_t[j], y_t[j]])
            if x_t[i]>=a and x_t[i]<=(l-a) and y_t[i]>=a and y_t[i]<=(l-a): 
                dist = distance.euclidean(X_i,X_j)
            else:
                offsets = np.array([[w, k] for w in [-1, 0, 1] for k in [-1, 0, 1]])
                X_j_offsets = X_j+offsets*l
                distances = distance.cdist([X_i], X_j_offsets)    
                dist = np.min(distances)
            if dist<=a:
                indice_dist[i,j] = 1
                indice_dist[j,i] = 1
    return indice_dist


def distance_indice_matrix_vectorized(x_t, y_t, l, N, a):
    # Créer une matrice de coordonnées
    coords = np.column_stack((x_t, y_t))

    # Créer une matrice 3D pour stocker toutes les distances possibles
    distances = np.zeros((N, N, 9))

    # Calculer les distances pour chaque offset
    offsets = np.array([[w, k] for w in [-1, 0, 1] for k in [-1, 0, 1]])
    for idx, offset in enumerate(offsets):
        coords_offsets = coords + offset * l
        distances[:, :, idx] = cdist(coords, coords_offsets)

    # Trouver la distance minimale pour chaque paire de particules
    min_distances = np.min(distances, axis=2)

    # Créer une matrice de 0 et de 1 en fonction de si la distance est inférieure à a ou pas
    indice_dist = (min_distances <= a).astype(int)

    # Les particules sont à une distance nulle d'elles-mêmes
    np.fill_diagonal(indice_dist, 1)

    return indice_dist


def test_distance():
    # Définir les valeurs d'entrée
    N = 10
    l = 10
    a = 1
    eta=0.1
    for i in tqdm(range(10)):
        x_t = np.random.uniform(0, l, N)
        y_t = np.random.uniform(0, l, N)
        theta_t = np.random.uniform(0, 2*np.pi, N)
        # Calculer les matrices de distances en utilisant les deux fonctions
        matrix1 = distance_indice_matrix(x_t, y_t, l, N, a).astype(int)
        matrix2 = distance_indice_matrix_vectorized(x_t, y_t, l, N, a).astype(int)
        if not np.array_equal(matrix1, matrix2):
            print("Les matrices ne sont pas égales !")
            print("Matrice 1 :")
            print(matrix1)
            print("Matrice 2 :")
            print(matrix2)
        
        lignes_i_1,collones_j_1 = np.where(matrix1==1)
        lignes_i_2,collones_j_2 = np.where(matrix2==1)
        if not np.array_equal(lignes_i_1, lignes_i_2):
            print("Les collones_i ne sont pas égales !")
            
        if not np.array_equal(collones_j_1, collones_j_2):
            print("Les collones_j ne sont pas égales !") 
        arg1 = matrix1
        arg2 = matrix2
        #Crétion de arg une matrice de taille NxN avec les angles de theta_t des particule qui sont à une distance inférieur à a
        arg1[lignes_i_1,collones_j_1] = theta_t[collones_j_1]
        arg2[lignes_i_2,collones_j_2] = theta_t[collones_j_2]
        if not np.array_equal(arg1, arg2):
            print("Les matrices arg ne sont pas égales !")
        #Création d'un vecteur contenant les moyennes sur les lignes de arg (sans les prendre en compte les 0)
        args_1 = np.ma.masked_equal(arg1, 0).mean(axis=1)
        args_2 = np.ma.masked_equal(arg2, 0).mean(axis=1)
        if not np.array_equal(args_1, args_2):
            print("Les vecteurs args ne sont pas égaux !")
        #Création de theta_tfut un vecteur de taille N avec la moyenne des angles de theta_t des particule qui sont à une distance inférieur à a
        theta_tfut_1 = args_1 
        theta_tfut_2 = args_2
        if not np.array_equal(theta_tfut_1, theta_tfut_2):
            print("Les vecteurs theta_tfut ne sont pas égaux !")
        #Calcul des positions à l'instant t+dt pour la particule i avec la condition aux bords
        x_tfut_1= (x_t + np.cos(theta_tfut_1)) % l
        x_tfut_2= (x_t + np.cos(theta_tfut_2)) % l
        if not np.array_equal(x_tfut_1, x_tfut_2):
            print("Les vecteurs x_tfut ne sont pas égaux !")
        
        y_tfut_2= (y_t + np.sin(theta_tfut_2)) % l 
        y_tfut_1= (y_t + np.sin(theta_tfut_1)) % l
        if not np.array_equal(y_tfut_1, y_tfut_2):
            print("Les vecteurs y_tfut ne sont pas égaux !")
        
        
    print("Les matrices sont égales !")    
test_distance()