# solver.py
import matlab.engine
import torch
import numpy as np
from config import T0, DT, X0, X1, N_POINTS

class MatlabPDESolver:
    def __init__(self, N_i):
        self.N_i = N_i
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        print("MATLAB engine started.")

    def solver(self, input_value, alpha):
        """
        Runs the MATLAB PDE solver on the provided initial condition(s).

        Parameters:
            input_value (np.ndarray or torch.Tensor): shape (N,) for a single case or (N, batch_size)
        Returns:
            np.ndarray: solution array of shape (N, batch_size)
        """
        
        if hasattr(input_value, 'cpu'):
            input_value = input_value.detach().cpu().numpy()
        else:
            input_value = np.array(input_value)

        if hasattr(alpha, 'cpu'):
            alpha = alpha.detach().cpu().numpy()
        else:
            alpha = np.array(alpha)

        dom = [X0, X1]
        N = N_POINTS
        x_vals = np.linspace(dom[0], dom[1], N)
        
        # Pass data to MATLAB workspace
        self.eng.workspace['x_vals'] = matlab.double(x_vals.tolist())
        self.eng.workspace['u_init_vals'] = matlab.double(input_value.tolist())
        self.eng.workspace['alpha_vals'] = matlab.double(alpha.tolist())
        
        # Start a parallel pool if available
        try:
            self.eng.eval("if exist('gcp','file')==2; if isempty(gcp('nocreate')); parpool; end; end", nargout=0)
        except matlab.engine.MatlabExecutionError:
            print("Parallel pool not started; Parallel Computing Toolbox may not be installed.")
        # Run the PDE solver in MATLAB using parfor
        self.eng.eval("""
        num_cases = size(u_init_vals,2);
        u = cell(1, num_cases);
        parfor i = 1:num_cases
            S_local = spinop([%d %d], [%d %f 1]);
            S_local.lin = @(u) alpha_vals(1,i) * diff(u,2) + 5*u;
            S_local.nonlin = @(u) -5*u.^3;
            S_local.init = chebfun(@(x) interp1(x_vals, u_init_vals(:,i), x, 'pchip'), [%d, %d], 'splitting', true, 'eps', 1e-6);
            u{i} = spin(S_local, %d, %f, 'plot', 'off');
        end
        """ % (X0, X1, T0, DT * self.N_i, X0, X1, N_POINTS, DT), nargout=0)
        
        # Extract final solution
        self.eng.eval("""
        u_final_vals = cell(1, num_cases);
        parfor i = 1:num_cases
            u_final_vals{i} = u{i}{2}(x_vals);
        end
        """, nargout=0)
        
        # Convert MATLAB cell array to a numpy array
        matlab_cell = self.eng.workspace['u_final_vals']
        results = [np.array(matlab_cell[i]).flatten() for i in range(len(matlab_cell))]
        results = np.stack(results, axis=-1)
        return results

    def bridge(self, pred_from_NN, alpha, device):
        """
        Bridge function to process network predictions before solving.
        Converts predictions to the proper shape and calls the solver.
        """
        # Convert predictions from shape (T-1, N, 1) to (N, T-1)
        to_solver = pred_from_NN.squeeze(-1).transpose(0, 1)
        alpha_to_solver = alpha.transpose(0, 1)
        from_solver = self.solver(to_solver, alpha_to_solver)
        
        from_solver_tensor = torch.from_numpy(from_solver).to(device=device, dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
        return from_solver_tensor

    def solver_test(self, input_value):
        """
        This is only for testing.
        """
        
        if hasattr(input_value, 'cpu'):
            input_value = input_value.detach().cpu().numpy()
        else:
            input_value = np.array(input_value)
        
        dom = [X0, X1]
        N = N_POINTS
        x_vals = np.linspace(dom[0], dom[1], N)
        
        self.eng.workspace['x_vals'] = matlab.double(x_vals.tolist())
        self.eng.workspace['u_init_vals'] = matlab.double(input_value.tolist())
        
        try:
            self.eng.eval("if exist('gcp','file')==2; if isempty(gcp('nocreate')); parpool; end; end", nargout=0)
        except matlab.engine.MatlabExecutionError:
            print("Parallel pool not started; Parallel Computing Toolbox may not be installed.")
   
        self.eng.eval("""
        num_cases = size(u_init_vals,2);
        u = cell(1, num_cases);
        parfor i = 1:num_cases
            S_local = spinop([%d %d], [%d %f 1]);
            S_local.lin = @(u) 0.0001 * diff(u,2) + 5*u;
            S_local.nonlin = @(u) -5*u.^3;
            S_local.init = chebfun(@(x) interp1(x_vals, u_init_vals(:,i), x, 'pchip'), [%d, %d], 'splitting', true, 'eps', 1e-6);
            u{i} = spin(S_local, %d, %f, 'plot', 'off');
        end
        """ % (X0, X1, T0, DT * self.N_i, X0, X1, N_POINTS, DT), nargout=0)
        
        self.eng.eval("""
        u_final_vals = cell(1, num_cases);
        parfor i = 1:num_cases
            u_final_vals{i} = u{i}{2}(x_vals);
        end
        """, nargout=0)
        
        matlab_cell = self.eng.workspace['u_final_vals']
        results = [np.array(matlab_cell[i]).flatten() for i in range(len(matlab_cell))]
        results = np.stack(results, axis=-1)
        return results

    def quit(self):
        self.eng.quit()
