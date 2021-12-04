import numpy as np


def log(data, filename):

    with open(filename, 'a') as f:
        np.savetxt(f, data, newline=' ', fmt='%0.6f')
        f.writelines('\n')
