NewPINNs, physics-informeing neural networks, a novel approach that couples neural networks and traditional solvers together to facilitate the learning of the underlying physics of differential equations.

This repository do an experiment for the inverse of the 1D linear ODE.

You may change the temporal, training, and the neural network parameters in the "config.py" file.

to run the experiment use:
python main.py --train --save --model_path your_desired_name.pth

Or run the experiment without saving the trained weights:
python main.py --train