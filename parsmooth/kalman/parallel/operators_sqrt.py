import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import vmap


def Tria(A):
    tria_A = jlinalg.qr(A.T, mode = 'economic')[1].T
    return tria_A


@vmap
def filtering_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        a_i, b_i, U_i, eta_i, Z_i
    elem2: tuple of array
        a_j, b_j, U_j, eta_j, Z_j

    Returns
    -------

    """
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
   
    
    return A, b, U, eta, Z


@vmap
def smoothing_operator(elem1, elem2):
    """
    Associative operator described in TODO: put the reference

    Parameters
    ----------
    elem1: tuple of array
        g_i, E_i, D_i
    elem2: tuple of array
        g_j, E_j, D_j

    Returns
    -------

    """
    g1, E1, D1 = elem1
    g2, E2, D2 = elem2

    
    g = E2 @ g1 + g2       #??formulas
    E = E2 @ E1              
    D = Tria(jnp.concatenate((E2 @ D1, D2), axis = 1))
    
    
    return g, E, D
