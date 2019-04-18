import numpy as np

def mean_and_error(mc_output,point_generation_method):
    """Computes an approximation to the root-mean-squared error in
    Monte-Carlo or Quasi-Monte Carlo approximations of expectations of
    quantities of interest.

    Inputs:

    mc_output - the output of generate_samples.generate_samples.

    point_generation_method - the argument 'point_generation_method'
    used in generate_samples.generate_samples to create mc_output.

    Outputs:

    list of length 2, [approx,error], where error is as in Kuo, et al
    (see below).

    """

    samples = mc_output[1]
    
    if point_generation_method is 'mc':

        num_qois = len(samples)

        N = len(samples[0])
        
        approx = []
        
        error = []
        
        for ii in range(num_qois):
            
            this_approx = samples[ii].mean()
            approx.append(this_approx)
            
            # Calculate the error - formula taken from
            # [Graham, Kuo, Nuyens, Scheichl, Sloan, JCP
            # 230, pp. 3668-3694 (2011), equation (4.4)]
            this_error = np.sqrt(((samples[ii] - this_approx)**2.0).sum()\
                                 /(float(N)*float(N-1)))
            error.append(this_error)

    elif point_generation_method == 'qmc':

        nu = len(samples)
        
        num_qois = len(samples[0])        
        
        approx_for_each_shift = [[] for ii in range(num_qois)]
                
        for ii in range(num_qois):

            for shift_no in range(nu):
                
                approx_for_each_shift[ii].append(samples[nu][ii].mean())


        approx_for_each_shift = [np.array(approximation) for approximation in approx_for_each_shift]
                
        # Calculate the QMC approximations for each qoi
        approx = [approximation.mean() for approximation in approx_for_each_shift]

        # Calculate the error for each qoi - formula taken from
        # [Graham, Kuo, Nuyens, Scheichl, Sloan, JCP
        # 230, pp. 3668-3694 (2011), equation (4.6)]
        error = [np.sqrt(((approx[ii]-approx_for_each_shift[ii])**2).sum()\
                         /(float(nu)*(float(nu)-1.0))) for ii in range(num_qois)]

    # Save data frame to file with extra metadata (how? -
    # utility function?)
    # TODO

    return [approx,error]
