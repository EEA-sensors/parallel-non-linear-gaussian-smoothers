# KF Library 0.05.0
# Created by Fatemeh Yaghoobi
# 08.09.2020 
# 25.01.2021 Parallel square-root kalman filter and RTS smoother are added.

import numpy as np
import scipy.linalg

############################################ Inverse ###########################################
inverse = lambda a: np.linalg.solve(a, np.eye(a.shape[0]))

######################################### Filtering Init #########################################
def filteringInitializer(F, Q, H, R, y, m0, P0, RunTime):
    n = int(2**np.ceil(np.log2(RunTime)))
    a = [[]] * n

    for k in range(0,RunTime):
        if k == 0:
            m1 = F @ m0
            P1 = F @ P0 @ F.T + Q
            S = H @ P1 @ H.T + R
            K = P1 @ H.T @ inverse(S)
            A = np.zeros(F.shape)
            b = m1 + K @ (y[:,0,None] - (H @ m1))
            C = P1 - (K @ S @ K.T)

#             eta = np.zeros((F.shape[0],1))
#             J = np.zeros(F.shape)
            eta = F.T @ H.T @ inverse(S) @ y[:,k,None]
            J = F.T @ H.T @ inverse(S) @ H @ F 
        else:
            S = H @ Q @ H.T + R
            K = Q @ H.T @ inverse(S)
            A = F - K @ H @ F
            b = K @ y[:,k,None]
            C = Q - K @ H @ Q

            eta = F.T @ H.T @ inverse(S) @ y[:,k,None]
            J = F.T @ H.T @ inverse(S) @ H @ F 

        a[k] = {'A': A, 'b': b, 'C': C, 'eta': eta, 'J': J}
    return a

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
    tria_A = scipy.linalg.qr(A.T , mode = 'economic')[1].T
    return tria_A

######################################### Square Filtering Init #########################################
def SqrFilteringInitializer(F, W, H, V, y, m0, N_0, RunTime):
    n = int(2**np.ceil(np.log2(RunTime)))
    a_sqr = [[]] * n
    
    ny = H.shape[0]
    nx = F.shape[0]

    for k in range(0,RunTime):
        if k == 0:
            m1 = F @ m0
            N1_ = Tria(np.concatenate((F@N_0, W), axis = 1))
            Psi_ = np.block([[H@N1_, V], [N1_, np.zeros((N1_.shape[0], V.shape[1]))]])
            Tria_Psi_ = Tria(Psi_)
            Psi11_ = Tria_Psi_[:ny , :ny]
            Psi21_ = Tria_Psi_[ny: ny + nx , :ny]
            Psi22_ = Tria_Psi_[ny: ny + nx , ny:]
            Y1 = Psi11_
            K1 = Psi21_ @ inverse(Psi11_)
            
            A = np.zeros(F.shape)
            b = m1 + K1 @ (y[:,0,None] - (H @ m1))
            U = Psi22_

#             eta = np.zeros((nx,1))
#             Z = np.zeros((nx,ny))
            Z1 = F.T @ H.T @ inverse(Y1).T
            eta = Z1 @ inverse(Y1) @ y[:,k,None]
            Z = np.block([Z1,np.zeros((nx,nx-ny))])
        
            

        else:
            Psi = np.block([ [ H @ W, V ] , [ W , np.zeros((nx,ny)) ] ])
            Tria_Psi = Tria(Psi)
            Psi11 = Tria_Psi[:ny , :ny]
            Psi21 = Tria_Psi[ny:ny + nx , :ny]
            Psi22 = Tria_Psi[ny:ny + nx , ny:]
            Y = Psi11
            K = Psi21 @ inverse(Psi11)
            A = F - K @ H @ F
            b = K @ y[:,k,None]
            U = Psi22
            
            Z1 = F.T @ H.T @ inverse(Y).T
            eta = Z1 @ inverse(Y) @ y[:,k,None]
            
            Z = np.block([Z1,np.zeros((nx,nx-ny))])
            
        a_sqr[k] = {'A': A, 'b': b, 'U': U, 'eta': eta, 'Z': Z}
    return a_sqr


######################################### Square Filtering PerSum #########################################
def SqrFiltering(a,b):
    c = {}
    nx = b['Z'].shape[0]
    ny = b['Z'].shape[1]
    
    Xi = np.block([[ a['U'].T @ b['Z'] , np.eye(a['U'].shape[1]) ], [ b['Z'] , np.zeros((b['Z'].shape[0], b['Z'].shape[0]))]])
    Tria_Xi = Tria(Xi)
    Xi11 = Tria_Xi[:a['U'].shape[1] , :a['U'].shape[1]]
    Xi21 = Tria_Xi[a['U'].shape[1]: a['U'].shape[1] + nx , :a['U'].shape[1]]
    Xi22 = Tria_Xi[a['U'].shape[1]: a['U'].shape[1] + nx , a['U'].shape[1]:]
    
    c['A'] = b['A'] @ a['A'] - b['A'] @ a['U'] @ inverse(Xi11).T @ Xi21.T @ a['A']
    c['b'] = b['A'] @ ( np.eye(a['U'].shape[0]) - a['U'] @ inverse(Xi11).T @ Xi21.T ) @ (a['b'] + a['U']@a['U'].T@b['eta']) + b['b']
    c['U'] = Tria(np.concatenate(( b['A']@a['U']@inverse(Xi11).T , b['U'] ), axis = 1))
    
    
    c['eta'] = a['A'].T @ ( np.eye(Xi21.shape[0]) - Xi21 @ inverse(Xi11) @ a['U'].T ) @ ( b['eta'] - b['Z'] @ b['Z'].T @ a['b'] ) + a['eta']
    c['Z'] = Tria(np.concatenate(( a['A'].T @ Xi22 , a['Z'] ), axis = 1))  
    c['Xi11'] = Xi11
    c['Xi21'] = Xi21
    c['Xi22'] = Xi22
    return c

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
#     c['L'] = b['E'] @ a['L'] @ b['E'] @ b['L'] ?????????????
    c['L'] = b['E'] @ a['L'] @ b['E'].T + b['L']
    
    return c
####################################### Square Smoothing init ####################################
def SqrSmoothingInitializer(F, W, m, N, RunTime):
    n = int(2**np.ceil(np.log2(RunTime)))
    a = [[]] * n
    nx = F.shape[0]

    for k in range(0,RunTime-1):
        
        Phi = np.block([[ F @ N[k] , W ], [ N[k] , np.zeros(( N[k].shape[0] , W.shape[1] )) ]])
        Tria_Phi = Tria(Phi)
        Phi11 = Tria_Phi[:nx , :nx]
        Phi21 = Tria_Phi[nx: nx + N[k].shape[0] , :nx]
        Phi22 = Tria_Phi[nx: nx + N[k].shape[0] , nx:]  
        
        E = Phi21 @ inverse(Phi11)
        D = Phi22
        g = m[k] - E @ F @ m[k]

        a[RunTime-k-1] = {'E': E, 'g': g, 'D': D, 'Phi11': Phi11}
        
    a[0] = {'E': np.zeros(F.shape), 'g': m[RunTime-1], 'D': N[RunTime-1], 'Phi11': np.zeros((nx,nx))}
        
    return a
###################################### Square Smoothing PerSum ##################################
def SqrSmoothing(a,b):
    c = {}
    
    c['E'] = b['E'] @ a['E']
    c['g'] = b['E'] @ a['g'] + b['g']
    c['D'] = Tria(np.concatenate((b['E'] @ a['D'], b['D']), axis = 1))
    
#     c['E'] = a['E'] @ b['E']
#     c['g'] = a['E'] @ b['g'] + a['g']
#     c['D'] = Tria(np.concatenate((a['E'] @ b['D'], a['D']), axis = 1))
    
    return c


#################################### parallel Scan Algorithm  #####################################
def parallelScanAlgorithm(a,RunTime, op):
    
    n = int(2**np.ceil(np.log2(RunTime)))
    a = a.copy()
    a0 = a.copy()
    
    ## Up pass    
    for d in range(0, int(np.log2(n)), 1):
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
    
    for d in range(int(np.log2(n)-1), -1, -1):
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



       