##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cmath import *
from scipy.spatial import distance
import matplotlib.animation as animation
from tqdm import tqdm

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

def distance_rnm_perio(x_n, y_n, x_m, y_m,l):
# Fonction qui calcule la distance entre les individus n et m
# x_n, y_n : coordonnées de l'individu n
# x_m, y_m : coordonnées de l'individu m
# l : longueur de la boite
    X_n = np.array([x_n, y_n])
    X_m = np.array([x_m, y_m])
    L = np.array([l, l])
    offsets = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
    X_m_offsets = X_m[None, :] + offsets @ L[:, None]
    distances = distance.cdist([X_n], X_m_offsets)
    return np.min(distances)

def distance_rnm(x_n, y_n, x_m, y_m,l):
# Fonction qui calcule la distance entre les individus n et m
# x_n, y_n : coordonnées de l'individu n
# x_m, y_m : coordonnées de l'individu m
# l : longueur de la boite
    X_n = np.array([x_n, y_n])
    X_m = np.array([x_m, y_m])
    return distance.euclidean(X_n, X_m)


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
    indices_rnm = np.eye(N)
    #calcul la distance matrice de taille NxN
    for i in range(N):
        args = 0
        for j in range(i+1,N):
            #Calcul de la distance entre les particules i et j
            # #si la particule est dans une boite de largeur l-2*a centré en (l/2,l/2)
            # if x_t[i]<a or y_t[i]<a or x_t[i]>l-a or y_t[i]>l-a:    
            #     dist = distance_rnm_perio(x_t[i], y_t[i], x_t[j], y_t[j],l)
            # else: 
            #     dist = distance_rnm(x_t[i], y_t[i], x_t[j], y_t[j],l)
            dist = distance_rnm_perio(x_t[i], y_t[i], x_t[j], y_t[j],l)
            if dist <= a:
                indices_rnm[i,j] = 1
                indices_rnm[j,i] = 1
    
    # Matrice de 0 et 1 pour chaque particule
    lignes_i,collones_j = np.where(indices_rnm==1) 
    arg = indices_rnm
    #Crétion de arg une matrice de taille NxN avec les angles de theta_t des particule qui sont à une distance inférieur à a
    arg[lignes_i,collones_j] = theta_t[collones_j]
    #Création d'un vecteur contenant les moyennes sur les lignes de arg (sans les prendre en compte les 0)
    args = np.ma.masked_equal(arg, 0).mean(axis=1)
    #Création de theta_tfut un vecteur de taille N avec la moyenne des angles de theta_t des particule qui sont à une distance inférieur à a
    theta_tfut = args + eta*np.random.uniform(-pi,pi,N)

    #Calcul des positions à l'instant t+dt pour la particule i avec la condition aux bords
    x_tfut= (x_t + v0*dt*np.cos(theta_tfut)) % l
    y_tfut= (y_t + v0*dt*np.sin(theta_tfut)) % l 

    return x_tfut, y_tfut,theta_tfut

def Solveur(N,l,a,v0,dt,eta,Nt):
# Fonction qui calcule les positions et directions des individus à chaque instant
###
# Entrées :
###
# N,l,a: nombre d'individus, longueur de la boite, échelle de longueur
# v0,dt,eta,Nt :vitesse, pas de temps, paramètre de bruit, nombre d'itérations
###
# Sortie : x_sol, y_sol, theta_sol : positions et directions à chaque instant matrice de taille Nt*N
    print("==================================")
    print("Simulation pour calculer la solution spacio-temporelle")
    x_t, y_t, theta_t = position_direction_init(N,l) #initialisation
    x_sol = np.zeros((Nt,N))
    y_sol = np.zeros((Nt,N))
    theta_sol = np.zeros((Nt,N))

    for i in tqdm(range(Nt-1)):
        x_t, y_t, theta_t = update_position_direction(N,l,a,v0,dt,eta,x_t,y_t,theta_t)
        #Ajout des vecteurs aux matrices
        x_sol[i,:] = x_t
        y_sol[i,:] = y_t
        theta_sol[i,:] = theta_t
    return x_sol , y_sol, theta_sol

def Calc_var_f(f,axis_nb):
    if axis_nb == 1:
        t = np.arange(0,len(f))
        var_f = np.zeros(len(f))
        for i in range(1,len(f)):
            var_f[i]=np.var(f[:(i)])
    if axis_nb == 2 :
        t = np.arange(0,np.shape(f)[0])
        var_f = np.zeros(np.shape(f)[0])
        for i in range(1,np.shape(f)[0]):
            var_f[i]=np.var(np.mean(f[:(i+1),:]-f[0,:],axis=1))
    return var_f,t

def Calc_mean_f(f,axis_nb):
    if axis_nb == 1:
        t = np.np.arange(0,len(f))
        mean_f = np.zeros(len(f))
        for i in range(1,len(f)):
            mean_f[i]=np.mean(f[:(i)])
    if axis_nb == 2 :
        t = np.arange(0,np.shape(f)[0])
        mean_f = np.zeros(np.shape(f)[0])
        for i in range(0,np.shape(f)[0]):
            mean_f[i]=np.mean(np.mean(f[:(i+1),:]-f[0,:],axis=1))
    return mean_f,t

def Calc_Phi(N,theta_t):
# Fonction qui calcule la moyenne des directions des individus à l'instant t
###
# Entrées : N : nombre d'individus, theta_t : Matrice de taille Nt*N
###
# Sortie : Phi_t, Var_phi_t : 
###
    phi = 1/N*np.abs(np.sum(np.exp(theta_t*1j),axis=1))
    Var_phi,t = Calc_var_f(phi,1)
    return phi, Var_phi,t

def AnimationGif(x_sol,y_sol,N,l,Nt,inter,fram,fig,scat,lines,name,Trajectoires,Save):
    #Définition de la fonction d'animation
    def anifunc(i):
    # Fonction animation qui trace les positions des individus à chaque instant et les trajectoire
    # !!!! Il faut avoir exécuté la fonction Solveur avant !!!!
        x_t = x_sol[i,:]
        y_t = y_sol[i,:]
        scat.set_offsets(np.c_[x_t, y_t])
        if Trajectoires==True:
            # Mettre à jour les données de chaque ligne
            for line, x, y in zip(lines, x_t, y_t):
                # Obtenir les données actuelles de la ligne
                old_x, old_y = line.get_data()
                # Si la ligne n'est pas vide et que la distance entre le nouveau point et le dernier point de la ligne est supérieure à l/2
                if len(old_x) > 0 and np.hypot(x - old_x[-1], y - old_y[-1]) > l/2:
                    # Réinitialiser la ligne
                    line.set_data([x], [y])
                else:
                    # Ajouter le nouveau point à la ligne
                    new_x = np.append(old_x, x)
                    new_y = np.append(old_y, y)
                    line.set_data(new_x, new_y)
        plt.xlim(0,l)
        plt.ylim(0,l)
        return lines + [scat]
    def init_func():
        # Effacer seulement les axes qui contiennent les lignes de trajectoire
        for line in lines:
            line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))  # Passer un tableau 2D vide à set_offsets
        return lines + [scat]
    ani = animation.FuncAnimation(fig, anifunc, frames=fram, interval=inter,repeat=True,init_func=init_func)
    namegif= name +".gif"
    print("==================================")
    print("Simulation pour créer un gif animé")  
    if Save==True :
        ani.save(namegif,writer="pillow")
        print('Gif animé sauvegardé :',namegif)
    print("Gif animé créé :",namegif)
    return ani
