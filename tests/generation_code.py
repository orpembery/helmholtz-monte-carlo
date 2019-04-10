from helmholtz_monte_carlo import error_analysis as err

def qmc_test(num_spatial_cores):
    """The test used to check parallelism is working correctly."""
    
    k = 5.0

    h_spec = (1.0,-1.5)

    J = 10

    nu = 20

    M = 5

    delta = 1.0

    lambda_mult = 1.0

    qois = ['integral','origin']

    return err.investigate_error(k,h_spec,J,nu,M,'qmc',delta,lambda_mult,qois,num_spatial_cores=num_spatial_cores,dim=2,display_progress=False)

def serial_filename():
    """Filename serial code is saved as."""
    return 'serial_code_output.pickle'
