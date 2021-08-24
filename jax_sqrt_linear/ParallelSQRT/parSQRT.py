#KF Library 0.05.0
# Created by Fatemeh Yaghoobi
# 08.09.2020 
# 25.01.2021 Parallel square-root kalman filter and RTS smoother are added.
# 14.08.2021  Parallel square-root kalman filter and RTS smoother in jax are added.

import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax.lax import associative_scan
from collections import namedtuple
from jax import vmap


c = namedtuple("c", ["A", "b", "U", "eta", "Z"] )
a_sqr = namedtuple("a_sqr", ['A', 'b', 'U', 'eta', 'Z'])
SqrSmoothingElements = namedtuple("SqrSmoothingElements", ['E', 'g', 'D'])
a = namedtuple("a", ['A', 'b', 'C', 'eta', 'J'])

######################################### Filtering Init #########################################
def filteringInitializer(F, Q, H, R, y, m0, P0, RunTime):

    for k in range(0,RunTime):
        if k == 0:
            m1 = F @ m0
            P1 = F @ P0 @ F.T + Q
            S = H @ P1 @ H.T + R
            K = jlinalg.solve(S.T, H @ P1.T).T
            A = np.zeros(F.shape)
            b = m1 + K @ (y[:,0,None] - (H @ m1))
            C = P1 - (K @ S @ K.T)

#             eta = np.zeros((F.shape[0],1))
#             J = np.zeros(F.shape)
            temp = jlinalg.solve(S.T, H @ F).T 
            eta = temp @ y[:,k,None]
            J = FHS_inv @ H @ F 
        else:
            S = H @ Q @ H.T + R
            K = jlinalg.solve(S.T, H @ Q.T).T
            A = F - K @ H @ F
            b = K @ y[:,k,None]
            C = Q - K @ H @ Q
            
            temp = jlinalg.solve(S.T, H @ F).T 
            eta = temp @ y[:,k,None]
            J = FHS_inv @ H @ F  

    return a(A, b, C, eta, J)

######################################### Filtering PerSum #########################################
def filtering(a,b):
    c = {}
    
    c['A'] = b['A'] @ inverse(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ a['A']
    c['b'] = b['A'] @ inverse(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ (a['b'] + a['C']@b['eta']) + b['b']
    c['C'] = b['A'] @ inverse(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ a['C']@b['A'].T + b['C']
    c['eta'] = a['A'].T @ inverse(np.eye(a['C'].shape[0]) + b['J']@a['C']) @ (b['eta'] - b['J']@a['b']) + a['eta']
    c['J'] =   a['A'].T @ inverse(np.eye(a['C'].shape[0]) + b['J']@a['C']) @ b['J']@a['A'] + a['J']
           
    return c

#########################################Triangularization operation###############################
def Tria(A):
    tria_A = jlinalg.qr(A.T, mode = 'economic')[1].T
    return tria_A



######################################### Square Filtering Init k=0 #########################################
def SqrFilteringInitializer_first_elem(F, W, H, V, m0, N_0, nx, ny, y):
    
    m1 = F @ m0
    N1_ = Tria(jnp.concatenate((F @ N_0, W), axis = 1))
    Psi_ = jnp.block([[H @ N1_, V], [N1_, jnp.zeros((N1_.shape[0], V.shape[1]))]])
    Tria_Psi_ = Tria(Psi_)
    Psi11_ = Tria_Psi_[:ny , :ny]
    Psi21_ = Tria_Psi_[ny: ny + nx , :ny]
    Psi22_ = Tria_Psi_[ny: ny + nx , ny:]
    Y1 = Psi11_
    K1 = jlinalg.solve(Psi11_.T, Psi21_.T).T

    A = jnp.zeros(F.shape)
    b = m1 + K1 @ (y - (H @ m1))
    U = Psi22_

    Z1 = jlinalg.solve(Y1, H @ F).T
    eta = jlinalg.solve(Y1.T, Z1.T).T @ y
    Z = jnp.block([Z1,jnp.zeros((nx,nx-ny))])
    
    return a_sqr(A[None,:,:], b[None,:,:], U[None,:,:], eta[None,:,:], Z[None,:,:])

######################################### Square Filtering Init k!=0 #########################################
def SqrFilteringInitializer_generic(F, W, H, V, nx, ny, y):
    
    Psi = jnp.block([ [ H @ W, V ] , [ W , jnp.zeros((nx,ny)) ] ])
    Tria_Psi = Tria(Psi)
    Psi11 = Tria_Psi[:ny , :ny]
    Psi21 = Tria_Psi[ny:ny + nx , :ny]
    Psi22 = Tria_Psi[ny:ny + nx , ny:]
    Y = Psi11
    K = jlinalg.solve(Psi11.T, Psi21.T).T
    A = F - K @ H @ F
    b = K @ y
    U = Psi22

    Z1 = jlinalg.solve(Y, H @ F).T
    eta = jlinalg.solve(Y.T, Z1.T).T @ y

    Z = jnp.block([Z1,jnp.zeros((nx,nx-ny))])
    
    return a_sqr(A, b, U, eta, Z)

######################################### Square Filtering Init #########################################
def SqrFilteringInitializer(F, W, H, V, m0, N_0, y):
    
    ny = H.shape[0]
    nx = F.shape[0]
    
    first_elem = SqrFilteringInitializer_first_elem(F, W, H, V, m0, N_0, nx, ny, y[0])
    generic_elems = vmap(lambda y: SqrFilteringInitializer_generic(F, W, H, V, nx, ny, y))(y[1:])
    

    
    return a_sqr(jnp.concatenate([first_elem.A, generic_elems.A],axis=0),
                 jnp.concatenate([first_elem.b, generic_elems.b],axis=0),
                 jnp.concatenate([first_elem.U, generic_elems.U],axis=0),
                 jnp.concatenate([first_elem.eta, generic_elems.eta],axis=0),
                 jnp.concatenate([first_elem.Z, generic_elems.Z],axis=0))

######################################### Square Filtering PerSum #########################################
@vmap
def SqrFiltering(elem1,elem2):
    
    A1, b1, U1, eta1, Z1 = elem1
    A2, b2, U2, eta2, Z2 = elem2
    nx = Z2.shape[0]
    ny = Z2.shape[1]
    
    Xi = jnp.block([[U1.T @ Z2 , jnp.eye(U1.shape[1])], [Z2 , jnp.zeros((nx, nx))]])
    Tria_Xi = Tria(Xi)
    Xi11 = Tria_Xi[:nx , :nx]
    Xi21 = Tria_Xi[nx: nx + nx , :nx]
    Xi22 = Tria_Xi[nx: nx + nx , nx:]
    
    A = A2 @ A1 - jlinalg.solve(Xi11, U1.T @ A2.T).T @ Xi21.T @ A1
    b = A2 @ ( jnp.eye(nx) - jlinalg.solve(Xi11, U1.T).T @ Xi21.T ) @ (b1 + U1 @ U1.T @ eta2) + b2
    U = Tria(jnp.concatenate(( jlinalg.solve(Xi11, U1.T @ A2.T).T , U2 ), axis = 1))
    
    
    eta = A1.T @ ( jnp.eye(Xi21.shape[0]) - jlinalg.solve(Xi11.T, Xi21.T).T @ U1.T ) @ ( eta2 - Z2 @ Z2.T @ b1 ) + eta1
    Z = Tria(jnp.concatenate(( A1.T @ Xi22 , Z1 ), axis = 1))  

    return c(A, b, U, eta, Z)

######################################### Smoothing init #########################################
def smoothingInitializer(F, Q, m, P, RunTime):
    n = int(2**np.ceil(np.log2(RunTime)))
    a = [[]] * n

    for k in range(0,RunTime-1):
        Pp  = F @ P[k] @ F.T + Q
        E = P[k] @ F.T @ inverse(Pp)
        g = m[k] - E @ F @ m[k]
        L = P[k] - E @ Pp @ E.T

        a[RunTime-k-1] = {'E': E, 'g': g, 'L': L}
        
    a[0] = {'E': np.zeros(F.shape), 'g': m[RunTime-1], 'L': P[RunTime-1]}
        
    return a

######################################### Smoothing PerSum #########################################
def smoothing(a,b):
    c = {}
    
    c['E'] = b['E'] @ a['E']
    c['g'] = b['E'] @ a['g'] + b['g']
    c['L'] = b['E'] @ a['L'] @ b['E'] @ b['L']
    
    return c

####################################### Square Smoothing init _ JAX ###################################
def SqrSmoothingInitializer_last_elements(m, N):
    return SqrSmoothingElements(jnp.zeros_like(N)[None,:,:], m[None,:,:], N[None,:,:])

def SqrSmoothingInitializer_generic_elements(F, W, m, N):
    nx = F.shape[0]
    Phi = jnp.block([[ F @ N , W ], [ N , jnp.zeros(( N.shape[0] , W.shape[1] )) ]])
    Tria_Phi = Tria(Phi)
    Phi11 = Tria_Phi[:nx , :nx]
    Phi21 = Tria_Phi[nx: nx + N.shape[0] , :nx]
    Phi22 = Tria_Phi[nx: nx + N.shape[0] , nx:]  

    E = jlinalg.solve(Phi11.T, Phi21.T).T
    D = Phi22
    g = m - E @ F @ m

    return SqrSmoothingElements(E, g, D)

def SqrSmoothingInitializer(F, W, m, N):
    
    last_elems = SqrSmoothingInitializer_last_elements(m[-1], N[-1])
    generic_elems = vmap(lambda m, N: SqrSmoothingInitializer_generic_elements(F, W, m, N))(m[:-1], N[:-1])
    
    return SqrSmoothingElements(jnp.concatenate([generic_elems.E, last_elems.E],axis=0),
                                jnp.concatenate([generic_elems.g, last_elems.g],axis=0),
                                jnp.concatenate([generic_elems.D, last_elems.D],axis=0))
  

###################################### Square Smoothing PerSum _ JAX ##################################
# @vmap
def SqrSmoothing(elem1,elem2):
    
    E1, g1, D1 = elem1 
    E2, g2, D2 = elem2
    
    E = E2 @ E1
#     E = E1 @ E2
    g = E2 @ g1 + g2
#     g = E1 @ g2 + g1
    D = Tria(jnp.concatenate((E2 @ D1, D2), axis = 1))
#     D = Tria(jnp.concatenate((E1 @ D2, D1), axis = 1))
    
    return SqrSmoothingElements(E, g, D)
#     return E, g, D, elem1, elem2

###################################### Square Smoothing PerSum _ JAX ##################################
@vmap
def SqrSmoothing_vmap(elem1,elem2):
    
    E1, g1, D1 = elem1 
    E2, g2, D2 = elem2
    
    E = E2 @ E1
#     E = E1 @ E2
    g = E2 @ g1 + g2
#     g = E1 @ g2 + g1
    D = Tria(jnp.concatenate((E2 @ D1, D2), axis = 1))
#     D = Tria(jnp.concatenate((E1 @ D2, D1), axis = 1))
    
    return SqrSmoothingElements(E, g, D)
#     return E, g, D, elem1, elem2

#################################### parallel Scan Algorithm  #####################################
def parallelScanAlgorithm(elems,RunTime, op):
    
    n = int(2**jnp.ceil(jnp.log2(RunTime)))
    
    a = [[]]*n
    for i in range(RunTime,-1,-1):
        a[i] = [elems.E[i], elems.g[i], elems.D[i]]
    a0 = a.copy()

    
    ## Up pass    
    for d in range(0, int(jnp.log2(n)), 1):
        for k in range(0, n, 2**(d+1)):
            i = k + 2**d - 1 
            j = k + 2**(d+1) - 1
            
            if len(a[j]) == 0:
                a[j] = a[i]
            elif len(a[i]) == 0:
                pass
            else:
                a[j] = op(a[i],a[j])
    a[-1] = []
    
    ## Down pass 
    
    for d in range(int(jnp.log2(n)-1), -1, -1):
        for k in range(0, n, 2**(d+1)): 
            i = k + 2**d - 1 
            j = k + 2**(d+1) - 1
            
            temp = a[i]
            a[i] =  a[j]
            
            if len(a[j]) == 0:
                a[j] = temp
            elif len(temp) == 0:
                pass
            else:
                a[j] = op(a[j],temp)
                
    ### Extra pass

    for k in range(1, n+1): 
        i = k-1
        
        if len(a[i]) == 0:
            a[i] = a0[i]
        elif len(a0[i]) == 0:
            pass
        else:
            a[i] = op(a[i],a0[i])
            
    a = a[:RunTime]
    
    return a

