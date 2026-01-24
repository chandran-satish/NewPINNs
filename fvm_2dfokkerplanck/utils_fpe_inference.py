import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pinn_fpe import ImagePredictorUNet

import fipy as fp



CONFIG_PATH = "./config_fpe_long.yaml"                  # <-- YAML config
CHECKPOINT_PATH = "./checkpoints/unet_Jan8/last.ckpt"  # <-- checkpoint
INPUT_VALUE = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
unet_config = config["model"]["unet_config"]
model = ImagePredictorUNet(config=unet_config)
model.to(DEVICE)
model.eval()
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
state_dict = {
    k.replace("model.", ""): v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("model.")
}
model.load_state_dict(state_dict, strict=True)

input_numpy = INPUT_VALUE*np.ones((32,32)).astype(np.float32)

# Torch expects: (batch, H, W)
x = torch.from_numpy(input_numpy).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    y = model(x)  # (1, 1, H, W)

output_numpy = y.squeeze().cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].imshow(input_numpy, cmap="viridis")
axs[0].set_title("Input")
axs[0].axis("off")

axs[1].imshow(output_numpy, cmap="viridis")
axs[1].set_title("UNet Output")
axs[1].axis("off")

plt.tight_layout()
#plt.savefig("unet_inference.png", dpi=200, bbox_inches="tight")
plt.close()


# Now let's plot the unet ouiput and the ground truth
mesh = fp.Grid2D(nx=32, ny=32, dx=1.0/32, dy=1.0/32)
x_coords, y_coords = mesh.cellCenters 
V = fp.CellVariable(mesh=mesh)
alpha = (2.0 - 1.0)*INPUT_VALUE + 1.0
V.setValue(alpha*np.sin(2*np.pi*x_coords)*np.sin(2*np.pi*y_coords))
D = 1.0
p_analytic = np.exp(-V.value / D)
p_analytic /= (p_analytic * mesh.cellVolumes).sum()
p_eq = p_analytic.reshape((32, 32), order='F')

nx = ny = 32
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

c0 = axs[0].contourf(X, Y, np.rot90(output_numpy, k=1), levels=100, cmap="jet")
c1 = axs[1].contourf(X, Y, np.rot90(p_eq, k=1),       levels=100, cmap="jet")

for ax in axs:
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

axs[0].set_title("Predicted Solution")
axs[1].set_title("True Solution")

# Shared colorbar: shrinks BOTH axes equally
fig.colorbar(c1, ax=axs, location="right", fraction=0.046, pad=0.04)

#plt.tight_layout()
#plt.savefig(f"fpe_contours_{INPUT_VALUE}.png", dpi=200, bbox_inches="tight")
plt.close()

# --- 3-panel contour plot: UNet, Ground Truth, Difference (UNet - GT)
# Rotate to fix swapped x/y (adjust k=1 vs k=-1 if needed)
unet = np.rot90(output_numpy, k=1)
gt   = np.rot90(p_eq,         k=1)
diff = unet-gt

nx = ny = 32
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals)

# Use identical contour levels for UNet and GT (recommended)
vmin = min(unet.min(), gt.min())
vmax = max(unet.max(), gt.max())
levels_main = np.linspace(vmin, vmax, 100)

# Symmetric levels for difference
dmax = np.max(np.abs(diff))
levels_diff = np.linspace(-dmax, dmax, 100)

fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# Panel 1: UNet
c_unet = axs[0].contourf(X, Y, unet, levels=levels_main, cmap="jet")
axs[0].set_title("Predicted Solution")
axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1)

# Panel 2: Ground truth
c_gt = axs[1].contourf(X, Y, gt, levels=levels_main, cmap="jet")
axs[1].set_title("True Solution")
axs[1].set_xlabel("x"); axs[1].set_ylabel("y")
axs[1].set_xlim(0, 1); axs[1].set_ylim(0, 1)

# Shared colorbar for panels 1–2 (keeps axes sizes consistent)
fig.colorbar(c_gt, ax=axs[:2], location="right", fraction=0.046, pad=0.04)

# Panel 3: Difference
c_diff = axs[2].contourf(X, Y, diff, levels=levels_diff, cmap="seismic")
axs[2].set_title("Error")
axs[2].set_xlabel("x"); axs[2].set_ylabel("y")
axs[2].set_xlim(0, 1); axs[2].set_ylim(0, 1)

# Colorbar for difference only
fig.colorbar(c_diff, ax=axs[2], location="right", fraction=0.046, pad=0.04)

plt.savefig(f"fpe_contours_{INPUT_VALUE}_diff.pdf", dpi=200, bbox_inches="tight")
plt.close()