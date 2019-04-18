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

    # Summing s,...,N+s-1, for s = 1,...,nu
    samples = [ [np.array(list(range(shift,int(N)+shift)),dtype='float') for qoi_index in range(2)] for shift in range(1,int(nu)+1)]
    
    mc_output = [1.0,samples]

    calc_answer = mean_and_error(mc_output,'qmc')

    true_mean = (N+nu)/2.0

    # Mean for s is (N+1)/2 + s - 1
    shift_means = (N + 1.0)/2.0 + np.array(range(1,int(nu)+1),dtype='float') - 1.0

    # Based on the formula referenced in the code itself
    true_error = np.sqrt(((true_mean - shift_means)**2.0).sum()/(nu*(nu-1.0)))
    
    assert np.isclose(calc_answer[0][0],true_mean)
    
    assert np.isclose(calc_answer[1][0],true_error)

    assert np.isclose(calc_answer[0][1],true_mean)
    
    assert np.isclose(calc_answer[1][1],true_error)



