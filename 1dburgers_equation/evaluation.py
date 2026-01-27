# evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Net
from config import T0, T1, DT_train, DT_solver, N_i, X0, X1, N_POINTS

def burgers_vectorized(u0, nu, dx, dt, nt):
    """
    Solve Burgers’ equation for multiple cases in parallel.
    
    Parameters
    ----------
    u0 : array_like, shape (ncases, nx)
        Initial velocity profiles for each case.
    nu : array_like, shape (ncases,)
        Viscosities for each case.
    dx : float
        Grid spacing.
    dt : float
        Time step.
    nt : int
        Number of time steps.
        
    Returns
    -------
    u : ndarray, shape (ncases, nx)
        Solution for each case at t = nt*dt.
    """
    # ensure arrays
    u = np.array(u0, dtype=float)
    nu = np.array(nu, dtype=float)

    # Broadcast nu to shape (ncases, 1) so it lines up with u[:, 1:-1]
    nu = nu[:, None]

    for n in range(nt):
        un = u.copy()
        # interior points: vectorized over all cases
        u[:, 1:-1] = (
            un[:, 1:-1]
            - un[:, 1:-1] * dt/(2*dx) * (un[:, 2:] - un[:, :-2])
            + nu * dt/dx**2 * (un[:, 2:] - 2*un[:, 1:-1] + un[:, :-2])
        )
        # boundary conditions (for all cases)
        u[:,  0] = 0
        u[:, -1] = 0

    return u

# Change the parameters below to see the results
alpha0 = 0.01
alpha1 = 0.01
Dalpha = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dx = 1.0/(N_POINTS-1)

def initial_condition(x):
    return -torch.sin(np.pi * x)

# path to the saves weights
MODEL_PATH = "weights_burgers_5000.pth"

# build grids
time_steps = torch.linspace(T0, T1, steps=round((T1 - T0) / DT_train) + 1, device=device)       # shape (T,)
T = len(time_steps)
xset   = torch.linspace(X0, X1, steps=N_POINTS, device=device)  # shape (Z,)
alphaset   = torch.linspace(alpha0, alpha1, steps=round((alpha1 - alpha0)/Dalpha) + 1, device=device)
R = len(alphaset)
nu_vals = np.array([alpha0])

t_grid, alpha_grid, x_grid = torch.meshgrid(time_steps,
                                             alphaset,
                                             xset,
                                             indexing='ij')
inputs = torch.stack((t_grid, alpha_grid, x_grid), dim=-1)
inputs_flat = inputs.view(-1, 3)

def evaluate(model_path: str):
    # load network
    net = Net()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()

    # compute true solution
    true_solutions = []
    current = initial_condition(xset.unsqueeze(1).to(device)).reshape(1,N_POINTS)
    true_solutions.append(current.cpu().numpy())

    for step in range(1, T):
        current_np = current.detach().cpu().numpy()
        next_np = burgers_vectorized(current_np, nu_vals, Dx, DT_solver, N_i)
        current = torch.from_numpy(next_np).to(device)
        true_solutions.append(current.cpu().numpy())

    # evaluate NN predictions for all (t, alpha, x)
    preds_all = net(inputs_flat).detach().cpu().numpy().reshape(T, R, N_POINTS)

    # prepare for plotting
    ncols = 4
    nrows = (T + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axs = axs.flatten()

    # x values on CPU
    x_cpu = xset.detach().cpu().numpy()

    for idx, t in enumerate(time_steps):
        # select predictions at time index idx and first alpha index 0
        preds_t = preds_all[idx, 0, :]
        true_data = true_solutions[idx]

        axs[idx].plot(x_cpu, preds_t, label="NN Output")
        axs[idx].plot(x_cpu, true_data, label="True", linestyle="--")
        axs[idx].set_title(f"t = {t:.2f}")
        axs[idx].set_xlabel("x")
        axs[idx].set_ylabel("u")
        axs[idx].legend()

    # turn off any unused subplots
    for j in range(idx+1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig("evaluation_plot.png")
    print("Saved plot to evaluation_plot.png")


if __name__ == "__main__":
    evaluate(MODEL_PATH)
