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
from IPython.display import HTML,display

##### Import files #####
from routines import *

################################### To change ###########################################
l=10                   #longueur de la boite                                             #
a=1                    #echelle de longueur                                              #
N=800                  #nombre de particules                                             #
v0 = 0.3              #Vitesse                                                           #
dt = 1                 #Pas de temps                                                     #
eta = 0.2              #eta : paramètre d'aléatoire                                      #
Nt = 100               #Nombre d'itérations                                              #
T = Nt*dt              #Temps final                                                      #
name = ("a="+str(a)+"_N="+str(N)
+"_eta="+str(eta)+"_v0="+str(v0)+"_Nt="+str(Nt) )  #Nom de la simulation                 #
Show_init = False      #True pour afficher les positions initiales dans un graphe        #
Animation = True       #True pour afficher l'animation                                   #
Save = True           #True pour sauvegarder l'animation                                #
Trajectoires = True   #True pour afficher les trajectoires (gérer pour N=1)
Show_Phi = True             #
#########################################################################################


##### Initialisation #####
x_init, y_init, theta_init = position_direction_init(N,l)
#Affichage des positions initiales
if Show_init==True:
    plt.figure()
    plt.scatter(x_init,y_init)
    plt.xlim(0,l)
    plt.ylim(0,l)
    plt.show()

###### Calcul de la solution spacio-temporelle #######

x_sol , y_sol, theta_sol = Solveur(N,l,a,v0,dt,eta,Nt) #Calcul des solutions


###### Mise en place de l'annimation #######

if Animation==True:
    ani = AnimationGif(x_sol,y_sol,N,l,Nt,100,name,Trajectoires,Save)
###### Calcul de Phi et affichage #######
phi,Var_phi = Calc_Phi(N,theta_sol)
t = np.arange(0,Nt)
if Show_Phi == True:
    plt.figure("Phi")
    plt.plot(t,phi,label="Phi");plt.plot(t,Var_phi,label="Variance de Phi")
    plt.legend();plt.xlim(0);plt.ylim(0);plt.xlabel("Temps");plt.ylabel("Phi");plt.show()