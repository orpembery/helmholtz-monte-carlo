from helmholtz_monte_carlo import error_analysis as err

k_range = [10.0]

h_spec = (1.0,-1.5)

J_range = [2]

nu = 2

M_range = [2]

delta = 2.0

lambda_mult = 1.0

qoi = 'integral'

err.investigate_error(k_range,h_spec,J_range,nu,M_range,'mc',delta,lambda_mult,qoi,dim=2)

err.investigate_error(k_range,h_spec,J_range,nu,M_range,'qmc',delta,lambda_mult,qoi,dim=2)
