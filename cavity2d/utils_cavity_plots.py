import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from utils_cavity_ngsolve import*
from pinn_cavity import FEMPhysicsModule

from scipy.interpolate import RegularGridInterpolator
from ngsolve import GridFunction, CoefficientFunction, NodeId, TaskManager
from ngsolve.fem import NODE_TYPE

def main():
    ckpt_path = "unet_checkpoints_May2_test2/last.ckpt"
    config_path = "config_cavity.yaml"
    ux, uy, pressure = get_unet_pred(ckpt_path, config_path, 2500)

    make_2dheatmap_matrix(ux, 'ux.png')
    make_2dheatmap_matrix(uy, 'uy.png')
    make_2dheatmap_matrix(pressure, 'press.png')
    make_2dheatmap_matrix(np.sqrt(ux**2 + uy**2), 'velmag.png')


    #_, _, _, gfu = run_ngsolve_plotty(u_init=np.float64(ux), v_init=np.float64(uy), p_init=np.float64(pressure), 
    #                                 nx=64, ny=64, nu=1/2500, uin_max=1.0, tau=0.001, t_iter=1)

    gfu =  get_gfu(u_init=np.float64(ux.T), v_init=np.float64(uy.T), p_init=np.float64(pressure.T), nx=64, ny=64, nu = 1.0/2500)
    
    save_plot_vel_heatmap(gfu, 'velmap.png')

    _, _, _, gfu2 = run_ngsolve_plotty(u_init=np.float64(ux), v_init=np.float64(uy), p_init=np.float64(pressure), 
                                         nx=32, ny=32, nu=1/2500, uin_max=1.0, tau=0.001, t_iter=250000)

    save_plot_vel_heatmap(gfu2, 'velmap_true.png')

def get_gfu(u_init=None, v_init=None, p_init=None, nx=32, ny=32, nu = 1.0/2500):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)

    num_elements = mesh.ne
    area_per_element = 1.0 / num_elements

    # Approximate grid resolution
    n_grid = int(np.sqrt(1.0 / area_per_element))  # assuming square domain
    print(n_grid)

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


    # Apply initial conditions if provided
    if u_init is not None and v_init is not None and p_init is not None:
        #ufunc = VoxelCoefficient((0,0), (1,1), u_init, linear=True)
        #vfunc = VoxelCoefficient((0,0), (1,1), v_init, linear=True)
        #pfunc = VoxelCoefficient((0,0), (1,1), p_init, linear=True)
        #gfu.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        #gfu.components[1].Set(pfunc)

        vec_voxel = CoefficientFunction((
                        VoxelCoefficient((0, 0), (1, 1), u_init.T, linear=True),
                        VoxelCoefficient((0, 0), (1, 1), v_init.T, linear=True))
                    )
        gfu.components[0].Set(vec_voxel)

        gfu.components[1].Set(VoxelCoefficient((0, 0), (1, 1), p_init, linear=True))

    return gfu


def run_ngsolve_plotty(u_init=None, v_init=None, p_init=None, nx=32, ny=32, nu=0.001, uin_max=1.0, tau=0.001, t_iter=1000):
    # Create unit square domain
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05)).Curve(3)

    num_elements = mesh.ne
    area_per_element = 1.0 / num_elements

    # Approximate grid resolution
    n_grid = int(np.sqrt(1.0 / area_per_element))  # assuming square domain
    print(n_grid)

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


    # Apply initial conditions if provided
    if u_init is not None and v_init is not None and p_init is not None:
        #ufunc = VoxelCoefficient((0,0), (1,1), u_init, linear=True)
        #vfunc = VoxelCoefficient((0,0), (1,1), v_init, linear=True)
        #pfunc = VoxelCoefficient((0,0), (1,1), p_init, linear=True)
        #gfu.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        #gfu.components[1].Set(pfunc)

        vec_voxel = CoefficientFunction((
                        VoxelCoefficient((0, 0), (1, 1), u_init.T, linear=True),
                        VoxelCoefficient((0, 0), (1, 1), v_init.T, linear=True))
                    )
        gfu.components[0].Set(vec_voxel)

        gfu.components[1].Set(VoxelCoefficient((0, 0), (1, 1), p_init, linear=True))

    # Apply lid motion
    lid_velocity = CoefficientFunction((uin_max, 0))
    gfu.components[0].Set(lid_velocity, definedon=mesh.Boundaries("top"))

    # Solve initial system
    #inv_stokes = a.mat.Inverse(X.FreeDofs())
    #res = f.vec - a.mat * gfu.vec
    #gfu.vec.data += inv_stokes * res

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
    U, V, P = sample_on_uniform_grid(gfu, nx=nx, ny=ny)
    return U, V, P, gfu

def make_2dheatmap_matrix(data, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('2D Heatmap')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.tight_layout()
    plt.savefig(filename)
    

def get_unet_pred(ckpt_path, config_path, Re):
    # Load and predict
    model = load_model(ckpt_path, config_path)
    ux, uy, pressure = predict_unet_output(model, Re = Re)
    return ux, uy, pressure

def get_true_pred():
    mesh, X, u, p, v, q, f, gfu = get_ngsolve_params()
    ux_true, uy_true, pressure_true, gfu = run_ngsolve_custom_params(nu = 1/2500.0, uin_max = 1.0, tau = 0.001, t_iter = 5000, 
                                            U_initial=np.zeros((32,32)), V_initial=np.zeros((32,32)), P_initial=np.zeros((32,32)), 
                                            mesh = mesh, X = X, u = u, p = p, v = v, q = q, f = f, gfu = gfu)
    save_plot_vel_heatmap(gfu, 'velmap_true.png')
    save_plot_pressure_heatmap(gfu, 'pmap_true.png')
    return ux_true, uy_true, pressure_true
    

# --- Load Model ---
def load_model(ckpt_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = FEMPhysicsModule(
        model_type=config['model']['type'],
        learning_rate=config['model']['learning_rate'],
        fem_iterations=config['training']['fem_iterations'],
        model_config=config['model']['unet_config']
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

# --- Predict UNet output ---
def predict_unet_output(model, Re=2500, device='cpu'):
    Re_norm = (Re - 2000) / (3000 - 2000)
    input_tensor = torch.ones((1, 32, 32)) * Re_norm
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0].cpu().numpy()

    return pred[0], pred[1], pred[2]  # ux, uy, pressure


# --- Plot heatmap from FEM GridFunction ---
def save_plot_vel_heatmap(gfu, filename):
        # Grid for evaluation
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    Vmag = np.zeros_like(X)
    velocity = gfu.components[0]

    for i in range(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            
            u_val, v_val = velocity(x, y)

            Vmag[i, j] = (u_val**2 + v_val**2)**0.5

    # Velocity magnitude heatmap
    plt.figure(figsize=(4, 4))
    contour = plt.contourf(X, Y, Vmag, levels=100, cmap='jet')
    plt.colorbar(contour, label='|u|')
    plt.title(f"Velocity Magnitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_plot_pressure_heatmap(gfu, filename):
    # Grid for evaluation
    nx, ny = 32, 32
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    P = np.zeros_like(X)
    pressure = gfu.components[1]

    for i in range(ny):
        for j in range(nx):
            x = x_vals[j]
            y = y_vals[i]
            
            P[i, j] = pressure(x, y)

    # Pressure heatmap
    plt.figure(figsize=(4,4))
    contour = plt.contourf(X, Y, P, levels=50, cmap='coolwarm')
    plt.colorbar(contour, label='Pressure')
    plt.title(f"Pressure Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main ---
if __name__ == "__main__":
    main()