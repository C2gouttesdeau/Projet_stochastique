####           Projet Stoch           #####
###########################################

##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import matplotlib.image as mpimg
import time
import warnings
warnings.filterwarnings("ignore")


##### Import files #####
from routines import *

################################### To change ###########################################
l=10                   #longueur de la boite                                             #
a=1                    #echelle de longueur                                              #
N=100                  #nombre de particules                                             #
v0 = 0.03              #Vitesse                                                          #
dt = 1                 #Pas de temps                                                     #
eta = 0.2              #eta : paramètre d'aléatoire                                      #
Nt = 50               #Nombre d'itérations                                              #
T = Nt*dt              #Temps final                                                      #
name = ("a="+str(a)+"_N="+str(N)
+"_eta="+str(eta)+"_v0="+str(v0)+"_Nt="+str(Nt) )  #Nom de la simulation                                          #
Show_init = False      #True pour afficher les positions initiales dans un graphe        #
Animation = True       #True pour afficher l'animation                                   #
Save = True           #True pour sauvegarder l'animation                                #
Trajectoires = False   #True pour afficher les trajectoires (gérer pour N=1)             #
#########################################################################################


##### Initialisation #####
x_init, y_init, theta_init = position_direction_init(N,l)
#Affichage des positions initiales
if Show_init==True:
    plt.figure(1)
    plt.scatter(x_init,y_init)
    plt.xlim(0,l)
    plt.ylim(0,l)
    plt.show()

###### Calcul de la solution spacio-temporelle #######

start_time = time.time()
x_sol , y_sol, theta_sol = Solveur(N,l,a,v0,dt,eta,Nt) #Calcul des solutions
end_time = time.time()
print("==================================")
print("Simulation pour calculer la solution spacio-temporelle")
print(f"Le temps calcul de la solution est {round(end_time-start_time,2)} secondes.")

###### Mise en place de l'annimation #######

if Animation==True:
    # Initialisation des paramètres
    fram = range(0,Nt)
    inter = 100
    fig = plt.figure()

    # Initialisation des positions et graphique
    x_t, y_t, theta_t = position_direction_init(N,l)
    scat = plt.scatter(x_t, y_t,marker='o',color='red',zorder=2,s=75)
    lines = [plt.plot([], [], lw=1,zorder=1)[0] for _ in range(x_sol.shape[1])]

    #Définition de la fonction d'animation (obligatoirement dans le main!!)
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
    ani = Gifanim(anifunc,fig,fram,inter,name,Save,init_func)
    plt.show()