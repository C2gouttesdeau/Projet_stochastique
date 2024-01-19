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
import os
import shutil

##### Import files #####
import routines

##### Simulation #####

for N,eta in zip(N,eta):
    print( )
    print("==================================")
    print("Simulation pour N = ",N," et eta = ",eta)
    print("==================================")

    size = (-(SizeMax-10)/Nmax)*N + SizeMax       #Taille des points 
    if N>Nmax : 
        size =10
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
    x_sol , y_sol, theta_sol,indice_stop  = routines.Solveur(N,l,a,v0,dt,eta,Nt,coef_dir_limit,Auto_stop,Perio) #Calcul des solutions
    if indice_stop < Nt-1:
        print("Convergence de la solution pour Nt=",indice_stop)
        Nt = indice_stop
        x_sol = x_sol[:Nt,:]
        y_sol = y_sol[:Nt,:]
        theta_sol = theta_sol[:Nt,:]
    name = ("a="+str(a)+"_N="+str(N)+"_eta="+str(eta)+"_v0="+str(v0)+"_Nt="+str(Nt)+"Perio_"+str(Perio) )  #Nom de la simulation
    
    #### Change de répertoire si sauvegarde ####
    if Save == True:
        if os.path.exists('Data/'+name):
            shutil.rmtree('Data/'+name)
        os.makedirs('Data/'+name)
        print('==================================')
        print("Sauvegarde dans le dossier : ",name)
    
    ###### Sauvegarde des données #######
    if Save == True:
        np.savetxt('Data/'+name+'/'+'x_sol_'+name+'.csv',x_sol) #ligne =temps collones=indices particules
        np.savetxt('Data/'+name+'/'+'y_sol_'+name+'.csv',y_sol)
        np.savetxt('Data/'+name+'/'+'theta_sol_'+name+'.csv',theta_sol)

    ###### Mise en place de l'annimation #######
    if Animation==True:
        # Initialisation des paramètres
        fram = range(0,Nt)
        fig = plt.figure("Animation",(9,9))
        # plt.close()
        # Initialisation graphique
        scat = plt.scatter([], [],marker='o',color='red',zorder=2,s=size)
        lines = lines = [plt.plot([], [], lw=2,zorder=1)[0]] #[plt.plot([], [], lw=1,zorder=1)[0] for _ in range(x_sol.shape[1])]
        ani = routines.AnimationGif(x_sol,y_sol,N,l,Nt,100,fram,fig,scat,lines,name,Trajectoires,Save,Perio)
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
            plt.savefig('Data/'+name+'/'+"Phi_"+name+".png")
        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()
        # #Calcul de la variance et affichage
        # var_xt,mean_varxt,tx = routines.Calc_var_vec(x_sol,dt)
        # var_yt,mean_varyt,ty = routines.Calc_var_vec(y_sol,dt)
        # plt.figure()
        # plt.plot(tx,var_xt,label="var_xt")
        # plt.plot(ty,var_yt,label="var_yt")
        # plt.plot(tx,mean_varxt,label="mean_varxt")
        # plt.plot(ty,mean_varyt,label="mean_varyt")
        # plt.xlim(0)
        # plt.xlabel("Temps")
        # plt.ylabel("Variance")
        # plt.legend()
        # if Save == True:
        #     plt.savefig("var_"+name+".png")
        # if Show_Analyse == True:
        #     plt.show()
        # else:
        #     plt.close()
        # #Calcul de la moyenne et affichage
        # mean_xt,mean_mean_xt,tx = routines.Calc_mean_vec(x_sol,dt)
        # mean_yt,mean_mean_yt,ty = routines.Calc_mean_vec(y_sol,dt)
        # plt.figure("mean")
        # plt.plot(tx,mean_xt,label="mean_xt")
        # plt.plot(ty,mean_yt,label="mean_yt")
        # plt.plot(tx,mean_mean_xt,label="mean_mean_xt")
        # plt.plot(ty,mean_mean_yt,label="mean_mean_yt")
        # plt.xlabel("Temps");plt.ylabel("Moyenne")
        # plt.xlim(0)
        # plt.legend()
        # if Save == True:
        #     plt.savefig("mean_"+name+".png")
        # if Show_Analyse == True:
        #     plt.show()
        # else:
        #     plt.close()
            
        #Calcul du déplacement et affichage
        mean_d,t = routines.Calc_deplacement(x_sol,y_sol,dt)
        plt.figure("Deplacement")
        plt.plot(t,mean_d)
        plt.xlabel("Temps");plt.ylabel("Deplacement")
        plt.xlim(0)
        if Save == True:
            plt.savefig('Data/'+name+'/'+"Deplacement_"+name+".png")
        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()
        #Calcul de la vitesse et affichage
        sum_ux,sum_uy,mean_somme_ux,mean_somme_uy,norm,mean_norm,dir,mean_dir,t = routines.Calc_somme_vec_vitesse(theta_sol,dt)
        plt.figure("Vitesse résulatante")
        plt.plot(t,sum_ux,label="sum_ux")
        plt.plot(t,sum_uy,label="sum_uy")
        plt.plot(t,mean_somme_ux,label="mean_sum_ux")
        plt.plot(t,mean_somme_uy,label="mean_sum_uy")
        plt.xlim(0)
        plt.xlabel("Temps");plt.ylabel("Somme vitesse")
        plt.legend()
        if Save == True:
            plt.savefig('Data/'+name+'/'+"Vitesse_"+name+".png")
        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()
        #Calcul barycentre et affichage
        x_bari,y_bari,mean_barycentre_x,mean_barycentre_y,n = routines.Calc_barycentre(x_sol,y_sol)
        plt.figure("barycentre")
        plt.plot(n,x_bari,label="x_bari")
        plt.plot(n,y_bari,label="y_bari")
        plt.plot(n,mean_barycentre_x,label="mean_x_bari")
        plt.plot(n,mean_barycentre_y,label="mean_y_bari")
        plt.xlabel("Particule N°");plt.ylabel("barycentre - position_initiale")
        plt.xlim(0)
        plt.legend()
        if Save == True:
            plt.savefig('Data/'+name+'/'+"barycentre_"+name+".png")
        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()
        #deplacement quadratique moyen
        deplacement_quad_moyen_x,tx = routines.deplacement_quad_mean(x_sol,dt)
        deplacement_quad_moyen_y,ty = routines.deplacement_quad_mean(y_sol,dt)

        coefs_Dx = np.polyfit(tx[:-1], deplacement_quad_moyen_x[:-1], 1)
        coefs_Dy = np.polyfit(ty[:-1], deplacement_quad_moyen_y[:-1], 1)
        p_Dx = np.poly1d(coefs_Dx)
        p_Dy = np.poly1d(coefs_Dy)
        equation_Dx = f"{coefs_Dx[0]:f}t + {coefs_Dx[1]:.5f}"
        equation_Dy = f"{coefs_Dy[0]:f}t + {coefs_Dy[1]:.5f}"

        plt.figure("Deplacement quadratique moyen")
        plt.plot(tx[:-1],deplacement_quad_moyen_x[:-1],label="deplacement_quad_moyen_x = "+equation_Dx)
        plt.plot(ty[:-1],deplacement_quad_moyen_y[:-1],label="deplacement_quad_moyen_y = "+equation_Dy)
        plt.xlabel("Temps");plt.ylabel("Deplacement quadratique moyen")
        plt.xlim(0)
        plt.legend()
        if Save == True:
            plt.savefig('Data/'+name+'/'+"Deplacement_quad_moyen_"+name+".png")
        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()

        #Coefficient de diffusion
        Dx,mean_Dx,tx = routines.Diff_coef(x_sol,dt)
        Dy,mean_Dy,ty = routines.Diff_coef(y_sol,dt)
        plt.figure("Coefficient de diffusion")
        plt.plot(t[:-1],Dx[:-1],label="Dx")
        plt.plot(t[:-1],mean_Dx*np.ones(len(t))[:-1],label="mean_Dx")
        plt.plot(ty[:-1],Dy[:-1],label="Dy")
        plt.plot(ty[:-1],mean_Dy*np.ones(len(ty))[:-1],label="mean_Dy")
        plt.xlabel("Temps");plt.ylabel("Coefficient de diffusion")
        plt.xlim(0)
        plt.legend()
        if Save == True:
            plt.savefig('Data/'+name+'/'+"Diffusion_"+name+".png")

        if Show_Analyse == True:
            plt.show()
        else:
            plt.close()
    
