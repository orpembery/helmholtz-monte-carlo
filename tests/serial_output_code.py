import generation_code as gen
import pickle

qmc_out = gen.qmc_test(1)

with open(gen.serial_filename(),'wb') as f:
            pickle.dump(qmc_out,f)
