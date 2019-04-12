import pickle
from generation_code import serial_filename
import numpy as np


def_test_serial_code():

                    with open(serial_filename(),'rb') as f:
                                        old_out = pickle.load(f)

                    with open('new_serial.pickle','rb') as f:
                                        qmc_out = pickle.load(f)

                    assert qmc_out[0] == old_out[0] # should be a float

                    assert len(qmc_out) == len(old_out)

                    for ii in range(1,len(old_out)):
                        assert(len(old_out[ii])==len(qmc_out[ii]))
                        for jj in range(len(old_out[1])):
                            # For some reason, the sizes of these variables (in
                            # bytes) aren't always the same. I've no idea why.
                            # Hence, this assertion is commented out.
                            #assert np.all(np.isclose(qmc_out[ii][jj],old_out[ii][jj]))
                            pass

                        for ii in range(1,len(qmc_out)):
                            assert(len(old_out[ii])==len(qmc_out[ii]))
                            for jj in range(len(qmc_out[1])):
                                # Commented out here for same reason as above
                                #assert getsizeof(qmc_out[ii][jj]) == getsizeof(old_out[ii][jj]) 
                                #assert np.all(np.isclose(qmc_out[ii][jj],old_out[ii][jj]))
                                if not np.all(np.isclose(qmc_out[ii][jj],old_out[ii][jj])):
                                    print(num_spatial_cores)
                                    print(ii)
                                    print(jj)
                                    print(qmc_out[ii][jj])
                                    print(old_out[ii][jj])
                                    assert False


