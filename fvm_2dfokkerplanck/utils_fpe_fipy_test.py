import fipy as fp
import matplotlib.pyplot as plt
import numpy as np

# 1. Mesh & Variables
L = 1.0
nx = 64
mesh = fp.Grid2D(nx=nx, ny=nx, dx=L/nx, dy=L/nx)

# Unpacking mesh.cellCenters returns NumPy arrays directly
x_coords, y_coords = mesh.cellCenters 

# 2. Define Potential V as a CellVariable
V = fp.CellVariable(mesh=mesh)
#V.setValue((x_coords-0.5)**2 + (y_coords - 0.5)**2)
V.setValue(np.sin(2*np.pi*x_coords)*np.sin(2*np.pi*y_coords))

# 3. Define Drift and PDE
drift_vector = V.faceGrad 
p = fp.CellVariable(mesh=mesh, value=1.0)
D = 1.0

#eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + fp.PowerLawConvectionTerm(coeff=drift_vector)
eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + fp.ExponentialConvectionTerm(coeff=drift_vector)

# 4. Solve for Equilibrium
dt = 1.0/64
steps = 256
for step in range(steps):
    eq.solve(var=p, dt=dt)

# 5. Ground Truth (Boltzmann Distribution): p ~ exp(-V/D)
# We normalize both so they are comparable
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

# Simulation Plot
im1 = ax1.pcolormesh(X, Y, Z_sim, shading='auto', cmap='viridis')
ax1.set_title("FiPy Numerical Equilibrium")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
plt.colorbar(im1, ax=ax1)

# Ground Truth Plot
im2 = ax2.pcolormesh(X, Y, Z_true, shading='auto', cmap='viridis')
ax2.set_title("Ground Truth (exp(-V/D))")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.colorbar(im2, ax=ax2)

# Ground Truth Plot
im3 = ax3.pcolormesh(X, Y, np.abs(Z_true - Z_sim), shading='auto', cmap='viridis')
ax3.set_title("Diff")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig("fpe_comparison.png")
print("Saved comparison plot to fpe_comparison.png")