# NewPINNs: Physics‑Informing Neural Networks Using Conventional Solvers

This repository contains the reference implementation for **NewPINNs**, a physics‑informing learning framework that couples neural networks with existing numerical solvers to solve differential equations.

NewPINNs departs from standard Physics‑Informed Neural Networks (PINNs) by removing PDE residuals from the loss function entirely. Instead, physical validity is enforced by directly embedding a numerical solver into the training loop and training the network through solver‑consistency.

Paper: https://arxiv.org/abs/2601.17207

## Citation

If you use this code or the NewPINNs framework in your research, please cite:

```
@misc{makki2026newpinnsphysicsinformingneuralnetworks,
      title={NewPINNs: Physics-Informing Neural Networks Using Conventional Solvers for Partial Differential Equations}, 
      author={Maedeh Makki and Satish Chandran and Maziar Raissi and Adrien Grenier and Behzad Mohebbi},
      year={2026},
      eprint={2601.17207},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.17207}, 
}
```
