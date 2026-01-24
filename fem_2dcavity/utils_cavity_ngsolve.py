import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ngsolve import*
from netgen.occ import*
from netgen.occ import X
from ngsolve import GridFunction, CoefficientFunction
from ngsolve import NodeId
from ngsolve.fem import NODE_TYPE

from netgen.geom2d import SplineGeometry
from netgen.geom2d import unit_square
from ngsolve import x, y

from ngsolve.internal import *  # For VoxelCoefficient

import multiprocessing as mp


def main():
    U = np.zeros((128, 64))
    V = np.zeros((128, 64))
    P = np.zeros((128, 64))
    #run_ngsolve_custom(nu = 0.001, uin_max = 1.5, tau = 0.001, t_iter = 500, U_initial = U, V_initial = V, P_initial = P)
    run_and_capture_ngsolve(nu = 0.001, uin_max = 1.5, tau = 0.001, t_iter = 500, U_initial = U, V_initial = V, P_initial = P)


def run_and_capture_ngsolve(nu = 0.001, uin_max = 1.0, tau = 0.001, t_iter = 1000, U_initial=None, V_initial=None, P_initial=None, tag=""):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=run_ngsolve_custom, args=(nu, uin_max, tau, t_iter, U_initial, V_initial, P_initial), kwargs={"conn": child_conn, "tag": tag})
    p.start()
    result = parent_conn.recv()  # blocks until child sends result
    p.join()

    U = result[0]
    V = result[1]
    P = result[2]
    gfu = result[3]

    return U, V, P, gfu


def run_ngsolve_custom(nu = 0.001, uin_max = 1.0, tau = 0.001, t_iter = 1000, U_initial=None, V_initial=None, P_initial=None, conn=None, tag=""):
    from ngsolve import SetNumThreads
    SetNumThreads(4)

    # Create unit square domain
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)

    # Define spaces
    V = VectorH1(mesh, order=3, dirichlet="top|bottom|left|right")
    Q = H1(mesh, order=2)
    X = V * Q

    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    stokes = (nu * InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q) * dx
    a = BilinearForm(stokes).Assemble()
    f = LinearForm(X).Assemble()

    gfu = GridFunction(X) # for boundaries 
    gfu_int = GridFunction(X) # for interior values

    # set gfu_int 
    if U_initial is not None and V_initial is not None and P_initial is not None:
        U_initial_64 = np.float64(U_initial)
        V_initial_64 = np.float64(V_initial)
        P_initial_64 = np.float64(P_initial)
        ufunc = VoxelCoefficient((0,0), (1,1), U_initial_64, linear=True)
        vfunc = VoxelCoefficient((0,0), (1,1), V_initial_64, linear=True)
        pfunc = VoxelCoefficient((0,0), (1,1), P_initial_64, linear=True)
        
        gfu_int.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        gfu_int.components[1].Set(pfunc)

    # Apply lid motion to gfu
    lid_velocity = CoefficientFunction((uin_max, 0))
    gfu.components[0].Set(lid_velocity, definedon=mesh.Boundaries("top"))

    # now loop through mesh and assign gfu with gfu_int values
    free_dofs = X.FreeDofs()
    for dof in range(X.ndof):
        if free_dofs[dof]:
            gfu.vec.data[dof] = gfu_int.vec[dof]
            
    #inv_stokes = a.mat.Inverse(X.FreeDofs())
    #res = f.vec - a.mat * gfu.vec
    #gfu.vec.data += inv_stokes * res

    mstar = BilinearForm(u * v * dx + tau * stokes).Assemble()
    inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    conv = BilinearForm(X, nonassemble=True)
    conv += (Grad(u) * u) * v * dx

    i = 0

    with TaskManager():
        while i < t_iter:
            res = conv.Apply(gfu.vec) + a.mat * gfu.vec
            gfu.vec.data -= tau * inv * res
            i += 1

    # Replace this with your own grid sampling function
    U, V, P = sample_on_uniform_grid(gfu, nx=32, ny=32)

    if conn is not None:
        conn.send((U, V, P, gfu))  # send tuple of arrays
        conn.close()
    #return U, V, P, gfu

def run_ngsolve_custom_old(nu = 0.001, uin_max = 1.0, tau = 0.001, t_iter = 1000, U_initial=None, V_initial=None, P_initial=None):
    # Create unit square domain
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)

    # Define spaces
    V = VectorH1(mesh, order=2, dirichlet="top|bottom|left|right")
    Q = H1(mesh, order=1)
    X = V * Q

    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    stokes = (nu * InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q) * dx
    a = BilinearForm(stokes).Assemble()
    f = LinearForm(X).Assemble()

    gfu = GridFunction(X)

    if U_initial is not None and V_initial is not None and P_initial is not None:
        U_initial_64 = np.float64(U_initial)
        V_initial_64 = np.float64(V_initial)
        P_initial_64 = np.float64(P_initial)
        ufunc = VoxelCoefficient((0,0), (1,1), U_initial_64.T, linear=True)
        vfunc = VoxelCoefficient((0,0), (1,1), V_initial_64.T, linear=True)
        pfunc = VoxelCoefficient((0,0), (1,1), P_initial_64.T, linear=True)
        gfu.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        gfu.components[1].Set(pfunc)

    # Apply lid motion
    lid_velocity = CoefficientFunction((uin_max, 0))
    gfu.components[0].Set(lid_velocity, definedon=mesh.Boundaries("top"))

    #inv_stokes = a.mat.Inverse(X.FreeDofs())
    #res = f.vec - a.mat * gfu.vec
    #gfu.vec.data += inv_stokes * res

    mstar = BilinearForm(u * v * dx + tau * stokes).Assemble()
    inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    conv = BilinearForm(X, nonassemble=True)
    conv += (Grad(u) * u) * v * dx

    i = 0

    with TaskManager():
        while i < t_iter:
            res = conv.Apply(gfu.vec) + a.mat * gfu.vec
            gfu.vec.data -= tau * inv * res
            i += 1

    # Replace this with your own grid sampling function
    U, V, P = sample_on_uniform_grid(gfu, nx=32, ny=32)
    return U, V, P, gfu

    
def run_ngsolve(nu=0.001, uin_max=1.0, tau=0.001, t_iter=1000):
    # Create unit square domain
    geo = SplineGeometry()
    p1 = geo.AppendPoint(0,0)
    p2 = geo.AppendPoint(1,0)
    p3 = geo.AppendPoint(1,1)
    p4 = geo.AppendPoint(0,1)
    geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="bottom")
    geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="right")
    geo.Append(["line", p3, p4], leftdomain=1, rightdomain=0, bc="top")
    geo.Append(["line", p4, p1], leftdomain=1, rightdomain=0, bc="left")
    mesh = Mesh(geo.GenerateMesh(maxh=0.05)).Curve(3)

    # Define spaces
    V = VectorH1(mesh, order=3, dirichlet="top|bottom|left|right")
    Q = H1(mesh, order=2)
    X = V * Q

    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    stokes = (nu * InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q) * dx
    a = BilinearForm(stokes).Assemble()
    f = LinearForm(X).Assemble()

    gfu = GridFunction(X)

    # Apply lid motion
    lid_velocity = CoefficientFunction((uin_max, 0))
    gfu.components[0].Set(lid_velocity, definedon=mesh.Boundaries("top"))


    inv_stokes = a.mat.Inverse(X.FreeDofs())
    res = f.vec - a.mat * gfu.vec
    gfu.vec.data += inv_stokes * res

    mstar = BilinearForm(u * v * dx + tau * stokes).Assemble()
    inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

    conv = BilinearForm(X, nonassemble=True)
    conv += (Grad(u) * u) * v * dx

    i = 0
    vel = gfu.components[0]

    with TaskManager():
        while i < t_iter:
            res = conv.Apply(gfu.vec) + a.mat * gfu.vec
            gfu.vec.data -= tau * inv * res
            i += 1
        
    # Replace this with your own grid sampling function
    U, V, P = sample_on_uniform_grid(gfu, nx=32, ny=32)
    return U, V, P, gfu

def sample_on_uniform_grid(gfu, nx=32, ny=32):
    """
    Sample the solution on a uniform grid
    
    Parameters:
    -----------
    gfu : GridFunction
        The solution containing velocity and pressure components
    nx, ny : int
        Number of points in x and y directions
    
    Returns:
    --------
    U, V, P : numpy.ndarray
        U and V velocity components and pressure on uniform grid
    """
    mesh = gfu.components[0].space.mesh
    
    # Create uniform grid points
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)
    
    # Initialize arrays
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    P = np.zeros((nx, ny))
    
    # Extract velocity and pressure components
    vel = gfu.components[0]
    pressure = gfu.components[1]
    
    # Sample at each grid point
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            point = mesh(x, y)
            vel_val = vel(point)
            U[i,j] = vel_val[0]  # x-velocity
            V[i,j] = vel_val[1]  # y-velocity
            P[i,j] = pressure(point)
            
    return U.T, V.T, P.T
    
if __name__ == "__main__":
   main()