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

##### Simulation #####

for N,eta in zip(N,eta):
    print( )
    print("==================================")
    print("Simulation pour N = ",N," et eta = ",eta)
    print("==================================")

    size = (-(SizeMax-10)/Nmax)*N + SizeMax       #Taille des points 
    ##### Initialisation #####
    x_init, y_init, theta_init = routines.position_direction_init(N,l)
    #Affichage des positions initiales
    if Show_init==True:
        plt.figure()
        plt.scatter(x_init,y_init,s=size)
        plt.xlim(0,l)
        plt.ylim(0,l)
        plt.show()

    ###### Calcul de la solution spacio-temporelle #######
    x_sol , y_sol, theta_sol,indice_stop  = routines.Solveur(N,l,a,v0,dt,eta,Nt,coef_dir_limit,Auto_stop) #Calcul des solutions
    if indice_stop < Nt-1:
        print("Convergence de la solution pour Nt=",indice_stop)
        Nt = indice_stop
        x_sol = x_sol[:Nt,:]
        y_sol = y_sol[:Nt,:]
        theta_sol = theta_sol[:Nt,:]
    name = ("a="+str(a)+"_N="+str(N)+"_eta="+str(eta)+"_v0="+str(v0)+"_Nt="+str(Nt) )  #Nom de la simulation
    if Save == True:
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
        lines = lines = [plt.plot([], [], lw=2,zorder=1)[0]] #[plt.plot([], [], lw=1,zorder=1)[0] for _ in range(x_sol.shape[1])]
        ani = routines.AnimationGif(x_sol,y_sol,N,l,Nt,100,fram,fig,scat,lines,name,Trajectoires,Save)
        if Show_Anim:
            plt.show()
        else:
            plt.close()
        
    ###### Annalyse ######
    if Analyse == True:
        #Calcul de Phi et affichage
        phi,Mean_phi,t = routines.Calc_Phi(theta_sol,N,dt)
        plt.figure("Phi")
        plt.plot(t[:-1],phi[:-1],label="Phi")
        plt.plot(t,Mean_phi,label="Mean de Phi")
        plt.xlim(0);plt.ylim(0,1)
        plt.xlabel("Temps");plt.ylabel("Phi")
        plt.legend()
        if Save == True:
            plt.savefig("Phi"+name+".png")
        
        #Calcul de la variance et affichage
        var_xt,tx = routines.Calc_var_vec(x_sol,dt)
        var_yt,ty = routines.Calc_var_vec(y_sol,dt)
        plt.figure();plt.plot(tx,var_xt,label="var_xt")
        plt.plot(ty,var_yt,label="var_yt")
        plt.xlim(0)
        plt.xlabel("Temps")
        plt.ylabel("Variance")
        plt.legend()
        if Save == True:
            plt.savefig("var"+name+".png")

        #Calcul de la moyenne et affichage
        mean_xt,mean_mean_xt,tx = routines.Calc_mean_vec(x_sol,dt)
        mean_yt,mean_mean_yt,ty = routines.Calc_mean_vec(y_sol,dt)
        plt.figure("mean")
        plt.plot(tx,mean_xt,label="mean_xt")
        plt.plot(ty,mean_yt,label="mean_yt")
        plt.plot(tx,mean_mean_xt,label="mean_mean_xt")
        plt.plot(ty,mean_mean_yt,label="mean_mean_yt")
        plt.xlabel("Temps");plt.ylabel("Moyenne")
        plt.xlim(0)
        plt.legend()
        if Save == True:
            plt.savefig("mean"+name+".png")

        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()

