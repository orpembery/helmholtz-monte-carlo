from helmholtz_monte_carlo import generate_samples as gen_samples

def qmc_test(num_spatial_cores,nearby_preconditioning=False,nearby_preconditioning_proportion=None):
    """The test used to check parallelism is working correctly."""
    
    k = 5.0

    h_spec = (1.0,-1.5)

    J = 10

    nu = 20

    M = 5

    delta = 1.0

    lambda_mult = 1.0

    j_scaling = 1.0

    qois = ['integral','origin']
    
    return gen_samples.generate_samples(k,h_spec,J,nu,M,'qmc',delta,lambda_mult,j_scaling,qois,num_spatial_cores=num_spatial_cores,dim=2,display_progress=True,physically_realistic=False,nearby_preconditioning=nearby_preconditioning,nearby_preconditioning_proportion=nearby_preconditioning_proportion)

def serial_filename():
    """Filename serial code is saved as."""
    return 'serial_code_output.pickle'
