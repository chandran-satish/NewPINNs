import fipy as fp
import matplotlib.pyplot as plt
import numpy as np

def main():
    nx = 32
    L = 1.0
    x_np = np.linspace(0, 1, nx)
    y_np = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x_np, y_np)
    u0 = np.exp(-(X**2 + Y**2) / 0.1) + np.random.rand()

    alpha = 1.0
    dt = 1.0/64
    num_iter = 4

    run_fipy_test(alpha = alpha, num_iter = num_iter, u0 = u0, nx = nx, L=L, dt = dt)

def run_fipy_test(alpha, num_iter, u0, nx = 64, L=1.0, dt = 1.0/64):
    mesh = fp.Grid2D(nx=nx, ny=nx, dx=L/nx, dy=L/nx)
    x_coords, y_coords = mesh.cellCenters 
    V = fp.CellVariable(mesh=mesh)
    V.setValue(alpha*np.sin(2*np.pi*x_coords)*np.sin(2*np.pi*y_coords))

    drift_vector = V.faceGrad 
    p = fp.CellVariable(mesh=mesh)
    p.setValue(u0.flatten(order='F'))

    D = 1.0
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + fp.ExponentialConvectionTerm(coeff=drift_vector)

    for step in range(num_iter):
        eq.solve(var=p, dt=dt)

    p_analytic = np.exp(-V.value / D)
    p_analytic /= (p_analytic * mesh.cellVolumes).sum()

    p_sim = p.value / (p.value * mesh.cellVolumes).sum()

    # 6. Plotting
    # Reshape using the coordinates to ensure the orientation is correct
    X = x_coords.reshape((nx, nx),order='F')
    Y = y_coords.reshape((nx, nx), order='F')
    Z_sim = p_sim.reshape((nx, nx), order='F')
    Z_true = p_analytic.reshape((nx, nx), order='F')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))


    im1 = ax1.pcolormesh(X, Y, Z_sim, shading='auto', cmap='viridis')
    ax1.set_title("FiPy Numerical Equilibrium")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1)


    im2 = ax2.pcolormesh(X, Y, Z_true, shading='auto', cmap='viridis')
    ax2.set_title("Ground Truth (exp(-V/D))")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.pcolormesh(X, Y, np.abs(Z_true - Z_sim), shading='auto', cmap='magma')
    ax3.set_title("Diff")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig("fpe_comparison2.png")
    print("Saved comparison plot to fpe_comparison2.png")

def run_fipy_custom(alpha, num_iter, p0, nx = 32, L=1.0, dt = 1.0/32):
    mesh = fp.Grid2D(nx=nx, ny=nx, dx=L/nx, dy=L/nx)
    x_coords, y_coords = mesh.cellCenters 
    V = fp.CellVariable(mesh=mesh)
    V.setValue(alpha*np.sin(2*np.pi*x_coords)*np.sin(2*np.pi*y_coords))

    drift_vector = V.faceGrad 
    p = fp.CellVariable(mesh=mesh)
    p.setValue(p0.flatten(order='F'))

    D = 1.0
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + fp.ExponentialConvectionTerm(coeff=drift_vector)

    for step in range(num_iter):
        eq.solve(var=p, dt=dt)

    p_sim = p.value / (p.value * mesh.cellVolumes).sum()

    Z_sim = p_sim.reshape((nx, nx), order='F')

    return Z_sim

if __name__ == "__main__":
    main()