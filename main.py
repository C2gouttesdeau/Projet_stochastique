####           Projet Stoch           #####
###########################################

##### Import libraries #####
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import matplotlib.image as mpimg

##### Import files #####
from routines import *

##### To change #####
l=10            #longueur de la boite
a=1             #echelle de longueur
N=50           #nombre de particules
v0 = 0.03      #Vitesse
dt = 1          #Pas de temps
eta = 0         #eta
Nt = 1       #Nombre d'itérations
T = Nt*dt       #Temps final
Simulation_name = ""
######################
##### Initialisation #####

#Initialisation des positions et directions
x_init, y_init, theta_init = position_direction_init(N,l)
#Affichage des positions initiales
Show = False
if Show==True:
    plt.figure(1)
    plt.scatter(x_init,y_init)
    plt.xlim(0,l)
    plt.ylim(0,l)
    plt.show()

#Histogramme
# plt.figure(1)
# plt.hist(x_init,y_init,width = 5, edgecolor = 'white')
# plt.show()

x_sol , y_sol, theta_sol = Solveur(N,l,a,v0,dt,eta,Nt) #Calcul des solutions

#Animation pour vérifier le déplacement des particules
Animation = False
Save = False
Trajectoires = False
Background = False
if Animation==True:
    fig, ax = plt.subplots()
    if Background == True : 
        # Charger l'image
        img = mpimg.imread('ciel.jpg')
        # Afficher l'image
        ax.imshow(img, extent=[0, l, 0, l])
    scat = ax.scatter(x_sol[0], y_sol[0],color='red',zorder=2)
    # Créer une ligne pour chaque particule
    lines = [ax.plot([x], [y], color='grey', linewidth=2, zorder=1)[0] for x, y in zip(x_sol[0], y_sol[0])]

    # Fixer l'échelle de l'axe
    ax.set_xlim([0, l])
    ax.set_ylim([0, l])
    
    def animate(i):
        scat.set_offsets(np.c_[x_sol[i], y_sol[i]])
        
        if Trajectoires==True:
            # Mettre à jour les données de chaque ligne
            for j in range(N):
                xdata, ydata = lines[j].get_data()
                xdata = np.append(xdata, x_sol[i][j])
                ydata = np.append(ydata, y_sol[i][j])
                lines[j].set_data(xdata, ydata)
            
    ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=50)
    if Save==True : 
        ani.save('animation_'+Simulation_name+'.gif', writer='pillow')
    plt.show()