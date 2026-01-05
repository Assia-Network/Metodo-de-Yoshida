import numpy as np
from numba import njit

#Función Hamiltoniano
def Hamiltoniano_potencial(r, p,masa,G_cnst,condicional="all"):   
    Ec=0
    if condicional=="cm" or condicional=="all":
        Ec=np.sum((np.linalg.norm(p, axis=2)**2)/(2*masa), axis=1)
    Ep=0
    if condicional=="pos" or condicional=="all":
        for g in range(masa.shape[0]-1):
            for m in range(g+1, masa.shape[0]):
                r_scalar=np.linalg.norm(r[:,g,:]-r[:,m,:], axis=1)    
                Ep+=-G_cnst*masa[g]*masa[m]/r_scalar                 
    return Ec, Ep

@njit
def fuerza_fun(data, G, masa):
    fuerza=np.zeros((data.shape[0],3))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j: 
                continue
            fuerza[i,:]+=G*masa[i]*masa[j]*(data[j,:]-data[i,:])/((np.linalg.norm(data[j,:]-data[i,:]))**3)
    return fuerza

coef_c = np.array([0.6756035959798289,-0.1756035959798288,-0.1756035959798288,0.6756035959798289])
coef_d = np.array([1.3512071919596578,-1.7024143839193153,1.3512071919596578])

@njit
def Yoshida(r,p,masa,G_cnst,dt_cnst,coef_c=coef_c,coef_d=coef_d):

    for i in range(3):
        r = r + coef_c[i]*dt_cnst*p/masa[:, None]
        p = p + coef_d[i]*dt_cnst*fuerza_fun(r,G_cnst,masa)

    r = r + coef_c[3]*dt_cnst*p/masa[:, None]

    return r,p
