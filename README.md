# porous_channel_ode

Here PINN solves the ODE:
    f'''' + Re(f'f'' - ff''') = 0
    
    Boundary conditions: 
        f'(1)  = 0 
        f(1)   = 1 
        f'(-1) = 0 
        f(-1)  = -1
        
For solving the ODE with the above boundary conditions, run train_pinn.py. Every 100th model will be saved in 'pinn_models/' folder. 
At the end of training, the best model is used for predicting the value of f, f'. u and v velocities can then be computed from f and f'.
