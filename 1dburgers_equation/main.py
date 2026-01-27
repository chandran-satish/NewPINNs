# main.py
import argparse
from experiment import PDEExperiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PDE Experiment")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--save", action="store_true", help="Save model weights after training")
    parser.add_argument("--model_path", type=str, default="pde_model_weights.pth", help="Path for saving/loading model weights")
    
    args = parser.parse_args()
    
    # Initialize PDEExperiment with solver only if not skipping.
    experiment = PDEExperiment()
    
    if args.train:
        experiment.train()
        if args.save:
            experiment.save_model(file_path=args.model_path)
    


