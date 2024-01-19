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

def distance_indice_matrix(x_t,y_t,l,N,a):
#Fonction qui renvoie une matrice de 0 et de 1
# 1 si les particule (i,j) de la matrices sont à une distance inférieur à a
# 0 sinon 
    indice_dist = np.zeros((N,N)) #les particules sont à une ditance nul d'elles même 
    L = np.array([l, l])
    for i in range(N):
        X_i= np.array([x_t[i], y_t[i]])
        for j in range(i+1,N):
            X_j = np.array([x_t[j], y_t[j]])
            # if x_t[i]>=a and x_t[i]<=(l-a) and y_t[i]>=a and y_t[i]<=(l-a): 
            #     dist = distance.euclidean(X_i,X_j)
            # else:
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
    indice_dist = (min_distances <= a)
    # Les particules sont à une distance nulle d'elles-mêmes
    np.fill_diagonal(indice_dist, True)

    return indice_dist#.astype(float)

def update_position_direction(N,l,a,v0,dt,eta,x_t,y_t,theta_t,Perio):
# Fonction qui calcule les positions et directions des individus à l'instant
###
# Entrées :
###
# N,l,a,v0,dt,eta: nombre d'individus, longueur de la boite, échelle de longueur, vitesse, pas de temps, paramètre de bruit
# x_t, y_t, theta_t : positions à l'instant t et directions à l'instant t vecteurs de taille N
###
# Sortie : x_tfut, y_tfut, theta_tfut : positions et directions à l'instant t+dt vecteurs de taille N
###
    if N==1 or a==0:
        indice_dist = np.zeros((N,N))
    else:
        indice_dist = distance_indice_matrix_vectorized(x_t,y_t,l,N,a)
    # Matrice de 0 et 1 pour chaque particule
    lignes_i,collones_j = np.where(indice_dist==True) 
    arg = indice_dist
    #Crétion de arg une matrice de taille NxN avec les angles de theta_t des particule qui sont à une distance inférieur à a
    arg[lignes_i,collones_j] = theta_t[collones_j]
    #Création d'un vecteur contenant les moyennes sur les lignes de arg (sans les prendre en compte les 0)
    args = np.ma.masked_equal(arg, False).mean(axis=1)
    args = np.ma.filled(args,0)
    #Création de theta_tfut un vecteur de taille N avec la moyenne des angles de theta_t des particule qui sont à une distance inférieur à a
    theta_tfut = args + eta*np.random.uniform(-pi,pi,N)
    #Calcul des positions à l'instant t+dt pour la particule i avec la condition aux bords
    if Perio == True:
        x_tfut= (x_t + v0*dt*np.cos(theta_tfut)) % l
        y_tfut= (y_t + v0*dt*np.sin(theta_tfut)) % l 
    else:
        x_tfut= (x_t + v0*dt*np.cos(theta_tfut))
        y_tfut= (y_t + v0*dt*np.sin(theta_tfut)) 

    return x_tfut, y_tfut,theta_tfut

def Solveur(N,l,a,v0,dt,eta,Nt,coef_dir_limit,Auto_stop,Perio):
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
    indice_stop = Nt-1

    for i in tqdm(range(Nt-1)):
        x_t, y_t, theta_t = update_position_direction(N,l,a,v0,dt,eta,x_t,y_t,theta_t,Perio)
        #Ajout des vecteurs aux matrices
        x_sol[i,:] = x_t
        y_sol[i,:] = y_t
        theta_sol[i,:] = theta_t
        if Auto_stop == True:
            if i>20 and i%20==0:
                phi,mean_phi,t = Calc_Phi(theta_sol,np.shape(theta_sol)[1],dt)
                if is_almost_constant(mean_phi, t, coef_dir_limit):
                    indice_stop = i
                    break
    return x_sol , y_sol, theta_sol, indice_stop

# def Calc_var_f(f,axis_nb):
#     if axis_nb == 1:
#         t = np.arange(0,len(f))
#         var_f = np.zeros(len(f))
#         for i in range(1,len(f)):
#             var_f[i]=np.var(f[:(i)])
#     if axis_nb == 2 :
#         t = np.arange(0,np.shape(f)[0])
#         var_f = np.zeros(np.shape(f)[0])
#         for i in range(1,np.shape(f)[0]):
#             var_f[i]=np.var(np.mean(f[:(i+1),:]-f[0,:],axis=1))
#     return var_f,t

def Calc_deplacement(x,y,dt):
    T = np.shape(x)[0]
    t = np.arange(0,T,dt)
    d = np.sqrt((x-x[0,:])**2+(y-y[0,:])**2) #distance entre t0 et t
    mean_d = np.mean(d,axis=1) # moyenne des déplacements pour l'ensemble des particules 
    return mean_d[:-1],t[:-1]

def Calc_somme_vec_vitesse(theta_sol,dt):
    t= np.arange(0,np.shape(theta_sol)[0],dt)
    ux = np.cos(theta_sol)
    uy = np.sin(theta_sol)

    sum_ux = np.sum(ux,axis=1)
    sum_uy = np.sum(uy,axis=1)
    norm = np.sqrt(sum_ux**2+sum_uy**2)
    dir = np.arctan(sum_uy/sum_ux)
    mean_somme_ux = np.zeros(len(t))
    mean_somme_uy = np.zeros(len(t))
    mean_norm = np.zeros(len(t))
    mean_dir = np.zeros(len(t))
    for i in range(len(t)):
        mean_somme_ux[i] = np.mean(sum_ux[:(i+1)])
        mean_somme_uy[i] = np.mean(sum_uy[:(i+1)])
        mean_norm[i] = np.mean(norm[:(i+1)])
        mean_dir[i] = np.mean(dir[:(i+1)])
    return sum_ux[:-1],sum_uy[:-1],mean_somme_ux[:-1],mean_somme_uy[:-1],norm[:-1],mean_norm[:-1],dir[:-1],mean_dir[:-1],t[:-1]

def Calc_var_vec(vec,dt):
    T = np.shape(vec)[0]
    t = np.arange(0,T,dt)
    # var_vec = np.zeros(len(t))
    # for i in range(len(t)):
    #     var_vec[i] = np.mean(np.var(vec[:(i+1),:]-vec[0,:],axis=0))
    var_vec = np.mean((vec-vec[0,:])**2, axis=1)
    # var_vec = np.mean(vec - np.mean(vec,axis=0)**2, axis=1)
    mean_var_vec = np.zeros(len(t))
    for i in range(len(t)):
        mean_var_vec[i] = np.mean(var_vec[:(i+1)])
    return var_vec[:-1],mean_var_vec[:-1],t[:-1]

def Calc_barycentre(x,y):
    N = np.shape(x)[1]
    n = np.arange(0,N)
    barycentre_x = np.mean(x,axis=0)-x[0,:]
    barycentre_y = np.mean(y,axis=0)-y[0,:]
    mean_barycentre_x = np.ones(N)*np.mean(barycentre_x)
    mean_barycentre_y = np.ones(N)*np.mean(barycentre_y)
    return barycentre_x,barycentre_y,mean_barycentre_x,mean_barycentre_y,n

def Diff_coef(x,dt):
    deplacement_quad_mean_x,t = deplacement_quad_mean(x,dt)
    D = deplacement_quad_mean_x[1:]/(2*t[1:])
    mean_D =np.mean(D)
    return D,mean_D,t[1:]

def deplacement_quad_mean(x,dt):
    t=np.arange(0,np.shape(x)[0],dt)
    deplacement_quad_moyen = np.zeros(len(t))
    for i in range(len(t)):
        deplacement_quad_moyen[i] = np.mean(np.mean((x[:i+1,:]-x[0,:])**2,axis=0))
    return deplacement_quad_moyen,t

def Calc_mean_vec(vec,dt):
    T = np.shape(vec)[0]
    t = np.arange(0,T,dt)
    mean_vec = np.mean(vec-vec[0,:],axis=1)
    # mean_vec = np.mean(vec-np.mean(vec,axis=0),axis=1)
    mean_mean_vec = np.zeros(len(t))
    for i in range(len(t)):
        mean_mean_vec[i] = np.mean(mean_vec[:(i+1)])
    return mean_vec[:-1],mean_mean_vec[:-1],t[:-1]

def Calc_Phi(theta_t,N,dt):
# Fonction qui calcule la moyenne des directions des individus à l'instant t
###
# Entrées : N : nombre d'individus, theta_t : Matrice de taille Nt*N
###
# Sortie : Phi_t, Var_phi_t : 
###
    T = np.shape(theta_t)[0]
    phi = 1/N*abs(np.sum(np.exp(theta_t*1j),axis=1))
    Mean_phi,t = [np.mean(phi[:(i+1)]) for i in range(len(phi))],np.arange(0,T,dt)
    return phi, Mean_phi,t

def AnimationGif(x_sol,y_sol,N,l,Nt,inter,fram,fig,scat,lines,name,Trajectoires,Save,Perio):
    #Définition de la fonction d'animation
    def anifunc(i):
    # Fonction animation qui trace les positions des individus à chaque instant et les trajectoire
    # !!!! Il faut avoir exécuté la fonction Solveur avant !!!!
        x_t = x_sol[i,:]
        y_t = y_sol[i,:]
        scat.set_offsets(np.c_[x_t, y_t])
        if Trajectoires==True:
            # Mettre à jour les données de la première ligne seulement
            line = lines[0]
            x = x_t[0]
            y = y_t[0]
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
        if Perio == True:
            plt.xlim(0,l)
            plt.ylim(0,l)
        else:
            plt.xlim(np.min(x_sol),np.max(x_sol))
            plt.ylim(np.min(y_sol),np.max(y_sol))
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

def is_almost_constant(phi, t, slope_threshold):
    # Ignorer les 20 premières valeurs
    phi = phi[20:]
    t = t[20:]

    # Effectuer une régression linéaire
    slope, intercept = np.polyfit(t, phi, 1)

    # Vérifier si le coefficient directeur est inférieur à la valeur spécifiée
    return np.abs(slope) <= slope_threshold