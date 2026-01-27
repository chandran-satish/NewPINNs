import matlab.engine
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- adjust these imports to your project structure ---
from model import Net    # your network class & IC function
from solver import MatlabPDESolver        # your Chebfun‑MATLAB solver wrapper
from config import T0, T1, DT, N_i, X0, X1, N_POINTS

alpha0 = 0.0001
alpha1 = 0.0001
Dalpha = 0.0001
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initial_condition(x):
    return x**2 * torch.cos(np.pi * x)

# path to your saved PyTorch model
MODEL_PATH = "weights_Allen–Cahn_10000_random_multi.pth"

# build grids
time_steps = torch.linspace(T0, T1, steps=round((T1 - T0) / DT) + 1, device=device)       # shape (T,)
T = len(time_steps)
xset   = torch.linspace(X0, X1, steps=N_POINTS, device=device)  # shape (Z,)
alphaset   = torch.linspace(alpha0, alpha1, steps=round((alpha1 - alpha0)/Dalpha) + 1, device=device)
R = len(alphaset)

# t_grid, alpha_grid, x_grid = torch.meshgrid(time_steps,
#                                              alphaset,
#                                              xset,
#                                              indexing='ij')
# inputs = torch.stack((t_grid, alpha_grid, x_grid), dim=-1)
# inputs_flat = inputs.view(-1, 3)
t_grid, alpha_grid = torch.meshgrid(time_steps,
                                             alphaset,
                                             xset,
                                             indexing='ij')
inputs = torch.stack((t_grid, alpha_grid), dim=-1)
inputs_flat = inputs.view(-1, 2)

# flatten for network evaluation
print(f"this is the input flat shape: {inputs_flat.shape}")


def evaluate(model_path: str):
    # load network
    net = Net()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()

    # set up Chebfun‑MATLAB solver
    solver = MatlabPDESolver(1)

    # compute true solution via Chebfun
    true_solutions = []
    current = initial_condition(xset.unsqueeze(1).to(device))
    true_solutions.append(current.cpu().numpy())

    for step in range(1, T):
        current_np = current.detach().cpu().numpy()
        next_np = solver.solver_test(current_np)
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
        axs[idx].plot(x_cpu, true_data, label="Chebfun True", linestyle="--")
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
