# EKFS Library 0.05.0
# Created by Fatemeh Yaghoobi
# 12.09.2020

import numpy as np


def f_turn(x,dt):
    
    x_k = np.zeros(x.shape)
    for i in range(x.shape[1]):
        w = x[-1,i]
        if w == 0:
            coswt = np.cos(w*dt)
            coswto = np.cos(w*dt)-1
            coswtopw = 0

            sinwt = np.sin(w*dt)
            sinwtpw = dt
        else:
            coswt = np.cos(w*dt)
            coswto = np.cos(w*dt)-1
            coswtopw = coswto/w

            sinwt = np.sin(w*dt)
            sinwtpw = sinwt/w

        F = np.array([[1, 0, sinwtpw,   -coswtopw, 0],
                      [0, 1, coswtopw,  sinwtpw,   0],
                      [0, 0, coswt,     sinwt,     0],
                      [0, 0, -sinwt,    coswt,     0],
                      [0, 0, 0,         0,         1]])
        x_k[:,i, None] = F @ x[:, i, None]
    
    return x_k

#######################################
def f_turn_dx(x,dt):

    w = x[-1]

    if w == 0:
        coswt = 1
        coswto = 0
        coswtopw = 0  

        sinwt = 0
        sinwtpw = dt

        dsinwtpw = 0
        dcoswtopw = -0.5*dt**2
    else:
        coswt = np.cos(w*dt)
        coswto = np.cos(w*dt)-1
        coswtopw = coswto/w  

        sinwt = np.sin(w*dt)
        sinwtpw = sinwt/w

        dsinwtpw = (w*dt*coswt - sinwt) / (w**2)
        dcoswtopw = (-w*dt*sinwt-coswto) / (w**2)

    df = np.zeros((len(x),len(x)))

    df[0,0] = 1
    df[0,2] = sinwtpw
    df[0,3] = -coswtopw
    df[0,4] = dsinwtpw * x[2] - dcoswtopw * x[3]

    df[1,1] = 1
    df[1,2] = coswtopw
    df[1,3] = sinwtpw
    df[1,4] = dcoswtopw * x[2] + dsinwtpw * x[3]

    df[2,2] = coswt
    df[2,3] = sinwt
    df[2,4] = -dt * sinwt * x[2] + dt * coswt * x[3]

    df[3,2] = -sinwt
    df[3,3] = coswt
    df[3,4] = -dt * coswt * x[2] - dt * sinwt * x[3]

    df[4,4] = 1

  
    return df

######################################### Filtering Init #########################################
def filteringInitializer(Q, R, y, f_fun, df_fun, h_fun, dh_fun, x_hat, m0, P0, RunTime):
    n = int(2**np.ceil(np.log2(RunTime)))
    a = [[]] * n
    
    

    for k in range(0,RunTime):
        
        
        if k == 0:
            F = df_fun(m0)
            
            m1 = f_fun(m0)
            P1 = F @ P0 @ F.T + Q
            
            H = dh_fun(m1)
            
            S = H @ P1 @ H.T + R
            K = P1 @ H.T @ np.linalg.inv(S)
            A = np.zeros(F.shape)
            b = m1 + K @ (y[:,0,None] - h_fun(m1))
            C = P1 - (K @ S @ K.T)

            eta = np.zeros((F.shape[0],1))
            J = np.zeros(F.shape)

        else:
            
            x_k_1 = x_hat[:,k-1,None]
            x_k = x_hat[:,k,None]
            F = df_fun(x_k_1)
            H = dh_fun(x_k)
            
            alpha = h_fun(x_k) + H @ f_fun(x_k_1) - H @ F @ x_k_1 - H @ x_k
            
            
            S = H @ Q @ H.T + R
            K = Q @ H.T @ np.linalg.inv(S)
            A = F - K @ H @ F
            b = K @ (y[:,k,None] - alpha)
            C = Q - K @ H @ Q

            eta = F.T @ H.T @ np.linalg.inv(S) @ (y[:,k,None] - alpha)
            J = F.T @ H.T @ np.linalg.inv(S) @ H @ F 

        a[k] = {'A': A, 'b': b, 'C': C, 'eta': eta, 'J': J}
    return a

######################################### Filtering PerSum #########################################

def filtering(a,b):
    c = {}
    
    c['A'] = b['A'] @ np.linalg.inv(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ a['A']
    c['b'] = b['A'] @ np.linalg.inv(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ (a['b'] + a['C']@b['eta']) + b['b']
    c['C'] = b['A'] @ np.linalg.inv(np.eye(a['C'].shape[0]) + a['C']@b['J']) @ a['C']@b['A'].T + b['C']
    c['eta'] = a['A'].T @ np.linalg.inv(np.eye(a['C'].shape[0]) + b['J']@a['C']) @ (b['eta'] - b['J']@a['b']) + a['eta']
    c['J'] =   a['A'].T @ np.linalg.inv(np.eye(a['C'].shape[0]) + b['J']@a['C']) @ b['J']@a['A'] + a['J']
           
    return c

######################################### Smoothing init #########################################

def smoothingInitializer(Q, f_fun, df_fun, x_hat, m, P, RunTime):
    n = int(2**np.ceil(np.log2(RunTime)))
    a = [[]] * n

    for k in range(0,RunTime-1):
        
        x_k = x_hat[:,k,None]
        F = df_fun(x_k)
        
        Pp  = F @ P[k] @ F.T + Q
        E = P[k] @ F.T @ np.linalg.inv(Pp)
        g = m[k] - E @ (f_fun(x_k) + F @ m[k] - F @ x_k)
        L = P[k] - E @ F @ P[k]

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


#################################### Parallel Scan Algorithm ######################################
#@jit
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




       