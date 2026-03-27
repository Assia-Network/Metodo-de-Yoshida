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

coef_c = np.array([0.6756035959798289,-0.1756035959798288,-0.1756035959798288,0.6756035959798289], dtype=np.float64)
coef_d = np.array([1.3512071919596578,-1.7024143839193153,1.3512071919596578], dtype=np.float64)

@njit
def Yoshida(r,p,masa,G_cnst,dt_cnst,coef_c=coef_c,coef_d=coef_d):

    num_etapas_p = len(coef_d)

    for i in range(num_etapas_p):
        r = r + coef_c[i]*dt_cnst*p/masa[:, None]
        p = p + coef_d[i]*dt_cnst*fuerza_fun(r,G_cnst,masa)

    r = r + coef_c[-1]*dt_cnst*p/masa[:, None]

    return r,p

def obtener_coeficientes_yoshida(grado):
    """
    Genera los coeficientes c y d para un orden 'grado' (debe ser par).
    Basado en la recursión S_{2n+2} = S_{2n}(z1) S_{2n}(z0) S_{2n}(z1)
    """
    # Empezamos con el orden 2 (Leapfrog/Verlet básico)
    c = np.array([0.5, 0.5], dtype=np.float64)
    d = np.array([1.0], dtype=np.float64)
    
    # Iteramos de 2 en 2 hasta llegar al grado deseado
    for n_actual in range(2, grado, 2):
        # n en la fórmula de Yoshida es (orden_actual / 2)
        n_formula = n_actual / 2
        
        # Cálculo de los factores x1 y x0 según Yoshida [cite: 164, 192]
        potencia = 1.0 / (n_actual + 1) # Es 1/(2n+1)
        x1 = 1.0 / (2.0 - 2.0**potencia)
        x0 = 1.0 - 2.0 * x1
        
        # Aplicar la triple composición: bloques de x1, luego x0, luego x1
        # Esto expande los arreglos c y d
        new_d = np.concatenate([x1 * d, x0 * d, x1 * d])
        
        # Para c, recordamos la fusión de los extremos (c_final + c_inicial) 
        c_centro_1 = x1 * c[-1] + x0 * c[0]
        c_centro_2 = x0 * c[-1] + x1 * c[0]
        
        new_c = np.concatenate([
            x1 * c[:-1], 
            [c_centro_1], 
            x0 * c[1:-1], 
            [c_centro_2], 
            x1 * c[1:]
        ])
        
        c, d = new_c, new_d
        
    return c, d