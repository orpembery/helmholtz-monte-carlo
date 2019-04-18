import numpy as np
from helmholtz_monte_carlo.calculate import mean_and_error

def test_mc_calculation():
    """Tests the mean and error calculation for monte-carlo-generated
    points works correctly.  """

    N = 10000.0
    
    samples = [np.array(list(range(1,int(N)+1)),dtype='float') for ii in range(2)]
    
    mc_output = [1.0,samples]

    calc_answer = mean_and_error(mc_output,'mc')

    true_mean = (N+1.0)/2.0

    true_error = np.sqrt(((true_mean - samples[0])**2.0).sum()/(N*(N-1.0)))
    
    assert calc_answer[0][0] == true_mean
    
    assert calc_answer[1][0] == true_error

    assert calc_answer[0][1] == true_mean
    
    assert calc_answer[1][1] == true_error


def test_qmc_calculation():
    """Tests the mean and error calculation for quasi-monte-carlo-generated
    points works correctly.  """

    N = 10000.0

    nu = 20.0
    
    samples = [[np.array(list(range(shift,int(N)+shift)),dtype='float')] for shift in range(int(nu))]
    
    mc_output = [1.0,samples]

    calc_answer = mean_and_error(mc_output,'qmc')

    true_mean = (N+nu)/2.0

    shift_means = (N + 2.0 * np.array(range(1,nu+1.0)) - 1.0)/2.0
    
    true_error = np.sqrt(((true_mean - shift_means)**2.0).sum()/(nu*(nu-1.0)))
    
    assert calc_answer[0][0] == true_mean
    
    assert calc_answer[1][0] == true_error

    assert calc_answer[0][1] == true_mean
    
    assert calc_answer[1][1] == true_error



