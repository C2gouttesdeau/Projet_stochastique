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
import cmath


##### Import files #####
from routines import *

################################### To change ###########################################
l=10                   #longueur de la boite                                             #
a=5                    #echelle de longueur                                              #
N=20                  #nombre de particules                                             #
v0 = 0.03              #Vitesse                                                          #
dt = 1                 #Pas de temps                                                     #
eta = 0.5              #eta : paramètre d'aléatoire                                      #
Nt = 10                #Nombre d'itérations                                              #
T = Nt*dt              #Temps final                                                      #
Simulation_name = "test"  #Nom de la simulation                                          #
Show_init = False      #True pour afficher les positions initiales dans un graph         #
Animation = True       #True pour afficher l'animation                                   #
Save = False           #True pour sauvegarder l'animation                                #
Trajectoires = True   #True pour afficher les trajectoires (gérer pour N=1)             #
Background = False     #True pour afficher le fond                                       #
Animation_infini =True #True pour afficher l'animation avec un nombre infini d'itérations#
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
#Calcul de la solution
if Animation_infini == False :
    start_time = time.time()
    x_sol , y_sol, theta_sol = Solveur(N,l,a,v0,dt,eta,Nt) #Calcul des solutions
    end_time = time.time()
    print(f"Le temps calcul de la solution est {round(end_time-start_time,2)} secondes.")

#Animation pour vérifier le déplacement des particules
if Animation==True:        
    start_time = time.time()
    fig, ax = plt.subplots()
    if Background == True : 
        # Charger l'image
        img = mpimg.imread('ciel.jpg')
        # Afficher l'image
        ax.imshow(img, extent=[0, l, 0, l])

    # Fixer l'échelle de l'axe
    ax.set_xlim([0, l])
    ax.set_ylim([0, l])

    if Animation_infini == True :
        x_t, y_t, theta_t = position_direction_init(N,l)
        scat = ax.scatter(x_t, y_t,marker='o',color='red',zorder=2,s=100)
        lines = [ax.plot([x], [y], color='grey', linewidth=2, zorder=1)[0] for x, y in zip(x_t, y_t)]
        def animate(i):
            global x_t,y_t,theta_t
            x_t_old, y_t_old = x_t.copy(), y_t.copy()  # Sauvegarder les positions précédentes
            x_t,y_t,theta_t = update_position_direction(N,l,a,v0,dt,eta,x_t,y_t,theta_t)
            scat.set_offsets(np.c_[x_t, y_t])
            
            if Trajectoires==True:
                # Mettre à jour les données de chaque ligne
                for j in range(N):
                    xdata, ydata = lines[j].get_data()
                    # Vérifier si la particule a traversé la boîte
                    if np.abs(x_t[j] - x_t_old[j]) < l / 2 and np.abs(y_t[j] - y_t_old[j]) < l / 2:
                        xdata = np.append(xdata, x_t[j])
                        ydata = np.append(ydata, y_t[j])
                        lines[j].set_data(xdata, ydata)
    else :
        scat = ax.scatter(x_sol[0], y_sol[0],marker='h',color='red',zorder=2,s=200)
        # Créer une ligne pour chaque particule
        lines = [ax.plot([x], [y], color='grey', linewidth=2, zorder=1)[0] for x, y in zip(x_sol[0], y_sol[0])]

        def animate(i):
            scat.set_offsets(np.c_[x_sol[i], y_sol[i]])
            
            if Trajectoires==True:
                # Mettre à jour les données de chaque ligne
                for j in range(N):
                    xdata, ydata = lines[j].get_data()
                    # Vérifier si la particule a traversé la boîte
                    if i > 0 and np.sqrt((x_sol[i][j] - x_sol[i-1][j]) ** 2 + (y_sol[i][j] - y_sol[i-1][j]) ** 2) < l / 2:
                        xdata = np.append(xdata, x_sol[i][j])
                        ydata = np.append(ydata, y_sol[i][j])
                        lines[j].set_data(xdata, ydata)
            
    ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=50)
    if Save==True : 
        ani.save('animation_'+Simulation_name+'.gif', writer='pillow')
    plt.show()
    end_time = time.time()
    print(f"Le temps de mise en place de l'annimation est {round(end_time-start_time,2)} secondes.")