import numpy as np
from numba import njit

#Función Hamiltoniano
@njit
def Hamiltoniano_potencial(r, p, masa, G_cnst, condicional="all"):   
    n_batch = r.shape[0]
    n_cuerpos = r.shape[1]
    
    Ec = np.zeros(n_batch)
    if condicional == "cm" or condicional == "all":
        for i in range(n_batch):
            suma_ec = 0.0
            for j in range(n_cuerpos):
                p_norm_sq = p[i, j, 0]**2 + p[i, j, 1]**2 + p[i, j, 2]**2
                suma_ec += p_norm_sq / (2.0 * masa[j])
            Ec[i] = suma_ec
            
    Ep = np.zeros(n_batch)
    if condicional == "pos" or condicional == "all":
        for i in range(n_batch):
            suma_ep = 0.0
            for g in range(n_cuerpos - 1):
                for m in range(g + 1, n_cuerpos):
                    dx = r[i, g, 0] - r[i, m, 0]
                    dy = r[i, g, 1] - r[i, m, 1]
                    dz = r[i, g, 2] - r[i, m, 2]
                    r_scalar = (dx**2 + dy**2 + dz**2)**0.5
                    suma_ep -= G_cnst * masa[g] * masa[m] / r_scalar
            Ep[i] = suma_ep
                
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