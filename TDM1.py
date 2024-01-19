import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fonctions

def tau_v(mp,nu,D_p):
    tau_v = mp/(3*np.pi*nu*D_p)
    return tau_v

def sigma(kB,T,mp):
    sigma = np.sqrt(3*kB*T/mp) 
    return sigma

def mp(D_p,rho_p):
    mp = 4/3*np.pi*(D_p/2)**3*rho_p
    return mp

def Du(sigma,tau_v):
    Du = (2*sigma**2)/tau_v
    return Du

def processus_vitesse(Du,tau_v,delta_t,u0):
    normale=np.random.normal(0,1,1) 
    u1 = u0*(1-delta_t/tau_v)+np.sqrt(Du*delta_t)*normale
    return u1

def position(x0,u0,delta_t):
    x1 = x0 + u0*delta_t
    return x1

def variance(L):
    moy = np.mean(L)
    var = np.mean([(x-moy)**2 for x in L])
    return var

nu = 10**(-6)
kB = 1.38064852*10**(-23)
T = 293

D_p = [10*10**(-6), 20*10**(-6), 30*10**(-6)]
#D_p = [20*10**(-6)] # m
D_p_notation = [r'$10^{-5}$', r'$2.10^{-5}$', r'$3.10^{-5}$']
#D_p_notation = [r'$2.10^{-5}$']
rho_p = 1000 # kg/m^3

#mp = 4/3*np.pi*(D_p/2)**3*rho_p
#
#tau_v = mp/(3*np.pi*nu*D_p)
#sigma = np.sqrt(3*kB*T/mp) 
#Du = (2*sigma**2)/tau_v
#
#var_th = sigma**2
#Ec_th = 3/2*kB*T
#print("var_th =", var_th)
#print("Ec_th =", Ec_th)
#
#print(tau_v)
#print(Du)
#print(sigma)
#
#delta_t = tau_v/10
#nb_iteration = int(100*tau_v/delta_t)

#nb_realisation = 100
nb_realisation = len(D_p)

L_var = [0 for k in range(nb_realisation)]
L_pos_fin = [[0,0,0] for k in range(nb_realisation)]

L_pos_x = [[] for k in range(nb_realisation)]
L_pos_y = [[] for k in range(nb_realisation)]
L_pos_z = [[] for k in range(nb_realisation)]

L_vit_x = [[] for k in range(nb_realisation)]
L_vit_y = [[] for k in range(nb_realisation)]
L_vit_z = [[] for k in range(nb_realisation)]

# Simulation
L_tauv = []

for real in range(nb_realisation):

    D_p_real = D_p[real]
    mp_real = mp(D_p_real,rho_p)
    tau_v_real = tau_v(mp_real,nu,D_p_real)
    L_tauv.append(tau_v_real)
    sigma_real = sigma(kB,T,mp_real)
    Du_real = Du(sigma_real,tau_v_real)

    #delta_t = tau_v_real/10
    delta_t = min([tau_v(mp(D_p[k],rho_p),nu,D_p[k]) for k in range(len(D_p))])
    #nb_iteration = int(100*tau_v_real/delta_t)
    nb_iteration = max([int(100*tau_v(mp(D_p[k],rho_p),nu,D_p[k])/delta_t) for k in range(len(D_p))])

    u0 = np.array([0.001])
    ux1 = [u0 for k in range(nb_iteration)]
    uy1 = [u0 for k in range(nb_iteration)]
    uz1 = [u0 for k in range(nb_iteration)]

    pos0 = np.array([0])
    x1 = [pos0 for k in range(nb_iteration)]
    y1 = [pos0 for k in range(nb_iteration)]
    z1 = [pos0 for k in range(nb_iteration)]

    for k in range(nb_iteration-1):
        ux1[k+1]=processus_vitesse(Du_real,tau_v_real,delta_t,ux1[k])
        uy1[k+1]=processus_vitesse(Du_real,tau_v_real,delta_t,uy1[k])
        uz1[k+1]=processus_vitesse(Du_real,tau_v_real,delta_t,uz1[k])

    for k in range(nb_iteration-1):
        x1[k+1]=position(x1[k],ux1[k],delta_t)
        y1[k+1]=position(y1[k],uy1[k],delta_t)
        z1[k+1]=position(z1[k],uz1[k],delta_t)

    L_pos_x[real] = x1
    L_pos_y[real] = y1
    L_pos_z[real] = z1

    L_vit_x[real] = ux1
    L_vit_y[real] = uy1
    L_vit_z[real] = uz1

    ux1_fluct = [ux1[k]-np.mean(ux1) for k in range(nb_iteration)]
    uy1_fluct = [uy1[k]-np.mean(uy1) for k in range(nb_iteration)]
    uz1_fluct = [uz1[k]-np.mean(uz1) for k in range(nb_iteration)]

    L_var[real] = np.var(ux1_fluct+uy1_fluct+uz1_fluct)
    L_pos_fin[real] = [x1[-1],y1[-1],z1[-1]]

    print("D_p =", D_p_real)

    var_th = sigma_real**2
    #Ec_th = 3/2*kB*T
    Ec_th = 0.5*mp_real*var_th 
    print("var_th =", var_th)
    print("Ec_th =", Ec_th)

    var_tot = L_var[real]  
    Ec_pr = 0.5*mp_real*var_tot
    print("var_pr =", var_tot)
    print("Ec_pr =", Ec_pr)


## On suppose le problème isotrope
#
#u_tot = ux1 #+uy1+uz1
#
## Statistiques
#
#plt.figure(0)
#plt.subplot(2,1,1)
#plt.plot(L_var,'+')
#plt.title("Variance des vitesses")
#plt.xlabel("Numéro de la réalisation")
#plt.ylabel("Variance")
#plt.subplot(2,1,2)
#plt.plot(L_var,'+')
#plt.title("Variance des vitesses")
#plt.xlabel("Numéro de la réalisation")
#plt.ylabel("Variance")
#
#
#var_tot = np.var(L_var)
#Ec_pr = 0.5*mp*var_tot
#print("var_pr =", var_tot)
#print("Ec_pr =", Ec_pr)
#
#L_dist = [np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2) for pos in L_pos_fin]
#moy_dist = np.mean(L_dist)
#print("moy_dist =", moy_dist)
#
#plt.figure(2)
#plt.plot(L_dist,'+')
#plt.xlabel("Numéro de la réalisation")
#plt.ylabel("Distance au point de départ de la particule")
#
## Courbes
#    
#plt.figure(1)
#plt.plot([k*delta_t for k in range(nb_iteration)],ux1)
#
#plt.figure(3)
#plt.hist(L_var, bins=50)
#plt.title("Histogramme des variances")
#plt.xlabel("Variance")
#plt.ylabel("Nombre de réalisations")
#
##plt.figure(4)
##plt.plot([k for i in range(nb_realisation)],L_dist,'+')
#
## Calcul de la pdf des vitesses et positions de la dernière réalisation
#
#plt.figure(5)
#plt.hist(ux1, bins=50)
#plt.title("Histogramme des vitesses selon x")
#plt.xlabel("Vitesse selon x")
#plt.ylabel("Nombre de réalisations")
#
#plt.figure(6)
#plt.hist(x1, bins=50)
#plt.title("Histogramme des positions selon x")
#plt.xlabel("Position selon x")
#plt.ylabel("Nombre de réalisations")
#
#f = Fitter(u_tot,distributions= ['norm'])
#f.fit()
#f.summary()
#

# Graphique 3D des trajectoires

plt.figure(7)
ax = plt.axes(projection='3d')
for real in range(nb_realisation):
    x1 = L_pos_x[real]
    y1 = L_pos_y[real]
    z1 = L_pos_z[real]
    ax.plot3D(x1,y1,z1,label=r'D_p ='+ str(D_p_notation[real])+ " m")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

plt.figure(8)
plt.plot([k*delta_t for k in range(len(x1))],x1)
plt.xlabel("Temps [s]")
plt.ylabel("Position [m]")
plt.title("Position projetée sur x")

print(D_p)
print(L_tauv)

plt.show()