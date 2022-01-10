import numpy as np
import matplotlib.pyplot as plt


def log(data, filename):

    with open(filename, 'a') as f:
        np.savetxt(f, np.array(data), newline=' ', fmt='%0.6f')
        f.writelines('\n')

def plot_meta(filename, title, y_lim, args, data_type='.png'):

    y = np.loadtxt(args.res_dir + '/' + filename)
    y = np.nan_to_num(y)

    plt.plot(np.array(range(len(y))), y)
    plt.title(title + ', $K={}$'.format(args.K))
    plt.ylim(y_lim)
    plt.savefig(args.res_dir + '/' + title + '_K' + str(args.K) + data_type, bbox_inches='tight')
    plt.close()


def plot_adpt(filename, title, y_lim, args, data_type='.png'):
    y = np.loadtxt(args.res_dir + '/' + filename)
    y = np.nan_to_num(y)
    for idx in range(0, y.shape[0], 100):
        plt.plot(np.array(range(y.shape[1])), y[idx])
    plt.legend(range(0, y.shape[0], 100))
    plt.title(title + ', $K={}$'.format(args.K))
    plt.ylim(y_lim)
    plt.savefig(args.res_dir + '/' + title + '_K' + str(args.K) + data_type, bbox_inches='tight')
    plt.close()

