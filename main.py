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
import routines
################################### To change ###########################################
l=10                   #longueur de la boite                                             #
a=1                    #echelle de longueur                                              #
N=10                 #nombre de particules                                             #
v0 = 0.3              #Vitesse                                                           #
dt = 1                 #Pas de temps                                                     #
eta = 0.2            #eta : paramètre d'aléatoire                                      #
Nt = 1000              #Nombre d'itérations                                              #
T = Nt*dt              #Temps final                                                      #
Show_init = False      #True pour afficher les positions initiales dans un graphe        #
Animation = True       #True pour afficher l'animation                                   #
Save = False           #True pour sauvegarder l'animation                                #
Trajectoires = False   #True pour afficher les trajectoires (gérer pour N=1)
Show_Phi = False             #
Nmax = 1000            #Nombre de particules max pourle calcul de size                #
size = (-190/Nmax)*N + 200       #Taille des points                      #
#########################################################################################

##### Initialisation #####
name = ("a="+str(a)+"_N="+str(N)+"_eta="+str(eta)+"_v0="+str(v0)+"_Nt="+str(Nt) )  #Nom de la simulation
x_init, y_init, theta_init = routines.position_direction_init(N,l)
#Affichage des positions initiales
if Show_init==True:
    plt.figure()
    plt.scatter(x_init,y_init,s=size)
    plt.xlim(0,l)
    plt.ylim(0,l)
    plt.show()

###### Calcul de la solution spacio-temporelle #######
x_sol , y_sol, theta_sol = routines.Solveur(N,l,a,v0,dt,eta,Nt) #Calcul des solutions
###### Mise en place de l'annimation #######
if Animation==True:
    # Initialisation des paramètres
    fram = range(0,Nt)
    fig = plt.figure("Animation")
    # plt.close()
    # Initialisation graphique
    scat = plt.scatter([], [],marker='o',color='red',zorder=2,s=75)
    lines = [plt.plot([], [], lw=1,zorder=1)[0] for _ in range(x_sol.shape[1])]
    ani = routines.AnimationGif(x_sol,y_sol,N,l,Nt,100,fram,fig,scat,lines,name,Trajectoires,Save)
    plt.show()
###### Calcul de Phi et affichage #######
phi,Var_phi,t = routines.Calc_Phi(N,theta_sol)
if Show_Phi == True:
    plt.figure("Phi")
    plt.plot(t,phi,label="Phi");plt.plot(t,Var_phi,label="Variance de Phi")
    plt.legend();plt.xlim(0);plt.ylim(0);plt.xlabel("Temps");plt.ylabel("Phi");plt.show()

var_xt,tx = routines.Calc_var_f(x_sol,2)
var_yt,ty = routines.Calc_var_f(y_sol,2)
var_theta,ttheta = routines.Calc_var_f(theta_sol,2)

plt.figure()
plt.plot(tx,var_xt,label="var_xt")
plt.plot(ty,var_yt,label="var_yt")
plt.plot(ttheta,var_theta,label="var_theta")
plt.legend()

mean_xt,tx = routines.Calc_mean_f(x_sol,2)
mean_yt,ty = routines.Calc_mean_f(y_sol,2)
mean_theta,ttheta = routines.Calc_mean_f(theta_sol,2)

plt.figure()
plt.plot(tx,mean_xt,label="mean_xt")
plt.plot(ty,mean_yt,label="mean_yt")
plt.plot(ttheta,mean_theta,label="mean_theta")
plt.legend()
plt.savefig("mean")

plt.show()

