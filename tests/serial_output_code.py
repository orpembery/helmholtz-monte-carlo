import generation_code as gen
import pickle

def serial_output_code():
    qmc_out = gen.qmc_test(1)

    with open(gen.serial_filename(),'wb') as f:
        pickle.dump(qmc_out,f)

if __name__ == '__main__':
    serial_output_code()
