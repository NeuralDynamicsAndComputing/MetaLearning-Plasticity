import numpy as np


def log(data, filename):

    with open(filename, 'a') as f:
        data_cpu = np.array(data)
        np.savetxt(f, data_cpu, newline=' ', fmt='%0.6f')
        f.writelines('\n')

