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
N=10             #nombre de particules                                             #
v0 = 0.3              #Vitesse                                                           #
dt = 1                 #Pas de temps                                                     #
eta = 0.2            #eta : paramètre d'aléatoire                                      #
Nt = 1000           #Nombre d'itérations                                              #
T = Nt*dt              #Temps final                                                      #
Show_init = False      #True pour afficher les positions initiales dans un graphe        #
Animation = True       #True pour afficher l'animation                                   #
Save = False           #True pour sauvegarder l'animation                                #
Trajectoires = False   #True pour afficher les trajectoires (gérer pour N=1)
Show_analyse = True             #
Nmax = 1000            #Nombre de particules max pourle calcul de size                #
SizeMax = 200
size = (-(SizeMax-10)/Nmax)*N + SizeMax       #Taille des points                        #
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
np.savetxt('x_sol_'+name+'.csv',x_sol) #ligne =temps collones=indices particules
np.savetxt('y_sol_'+name+'.csv',y_sol)
np.savetxt('theta_sol_'+name+'.csv',theta_sol)

###### Mise en place de l'annimation #######
if Animation==True:
    # Initialisation des paramètres
    fram = range(0,Nt)
    fig = plt.figure("Animation",(9,9))
    # plt.close()
    # Initialisation graphique
    scat = plt.scatter([], [],marker='o',color='red',zorder=2,s=size)
    lines = [plt.plot([], [], lw=1,zorder=1)[0] for _ in range(x_sol.shape[1])]
    ani = routines.AnimationGif(x_sol,y_sol,N,l,Nt,100,fram,fig,scat,lines,name,Trajectoires,Save)
    plt.show()

###### Annalyse ######
if Show_analyse == True:
    #Calcul de Phi et affichage
    phi,Mean_phi,t = routines.Calc_Phi(theta_sol,N,T,dt)
    plt.figure("Phi")
    plt.plot(t,phi,label="Phi")
    plt.plot(t,Mean_phi,label="Mean de Phi")
    plt.xlim(0);plt.ylim(0)
    plt.xlabel("Temps");plt.ylabel("Phi")
    plt.legend()
    plt.savefig("Phi"+name+".png")
    
    #Calcul de la variance et affichage
    var_xt,tx = routines.Calc_var_vec(x_sol,T,dt)
    var_yt,ty = routines.Calc_var_vec(y_sol,T,dt)
    plt.figure();plt.plot(tx,var_xt,label="var_xt")
    plt.plot(ty,var_yt,label="var_yt")
    plt.xlim(0);plt.ylim(0)
    plt.xlabel("Temps")
    plt.ylabel("Variance")
    plt.legend()
    plt.savefig("var"+name+".png")

    #Calcul de la moyenne et affichage
    mean_xt,mean_mean_xt,tx = routines.Calc_mean_vec(x_sol,T,dt)
    mean_yt,mean_mean_yt,ty = routines.Calc_mean_vec(y_sol,T,dt)
    plt.figure("mean")
    plt.plot(tx,mean_xt,label="mean_xt")
    plt.plot(ty,mean_yt,label="mean_yt")
    plt.plot(tx,mean_mean_xt,label="mean_mean_xt")
    plt.plot(ty,mean_mean_yt,label="mean_mean_yt")
    plt.xlabel("Temps");plt.ylabel("Moyenne")
    plt.xlim(0);plt.ylim(0)
    plt.legend()
    plt.savefig("mean"+name+".png")

    plt.show()

