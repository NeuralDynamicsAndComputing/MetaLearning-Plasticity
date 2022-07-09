import numpy as np
import matplotlib.pyplot as plt


def cumsum_sma(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]

    return ret[period - 1:] / period


def plot(res_dir, res_file, title, ylim=None, N=10000, moving_avg=True, period=17):

    y = np.nan_to_num(np.loadtxt(res_dir + '/' + res_file))[:N]

    if moving_avg:
        z = cumsum_sma(y, period)
        plt.plot(np.array(range(len(z))) + int((period - 1) / 2), z)
    else:
        plt.plot(range(len(y)), y)

    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([0, N])
    plt.show()


def plots(res_dir, res_file, title, ylim=None, N=10000, moving_avg=True, period=17):

    y = np.nan_to_num(np.loadtxt(res_dir + '/' + res_file))[:N]

    cmap = plt.get_cmap("tab10")
    print(y.shape[1])
    for idx in range(y.shape[1]):
        # -- moving average
        if moving_avg:
            z = cumsum_sma(y[:, idx], period)
            plt.plot(np.array(range(len(z))) + int((period - 1) / 2), z, color=cmap(idx))
        else:
            plt.plot(range(len(y)), y[:, idx], color=cmap(idx), alpha=0.5)

    plt.title(title)
    # plt.legend(['e_0', 'e_1', 'e_2', 'e_3'])
    plt.legend(['$W_{0,1}$', '$W_{1,2}$', '$W_{2,3}$', '$W_{3,4}$', '$W_{4,5}$'])
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([0, N])
    plt.show()

N = 10000

# ['fix_12_MX/', 'fix_MX/', 'fix_16_MX/', 'sym_MX/', 'fix_12_16_MX/']

for test_name in ['fix_FA/', 'fix_FA_16_b/' ]:
    res_dir = './tests/' + test_name

    # plots(res_dir, 'ort.txt', test_name[:-4] + ' orthogonality error', [0, 100], N=N, period=17)

    # plot(res_dir, 'loss.txt', test_name + '_loss', [0, 10], N=N, period=11)
    # plot(res_dir, 'acc.txt', test_name + '_accuracy', [0, 1], N=N, period=11)
    # plots(res_dir, 'e_ang_meta.txt', test_name + '_angle', [0, 120], N=N, period=11)
    # plots(res_dir, 'y_norm_meta.txt', test_name + '_y_norm', [0, 20], N=10000, period=17)
    # plots(res_dir, 'e_norm_meta_{}.txt'.format(title), title + '_e_norm', [0, 1], N=10000, period=17)
    plots(res_dir, 'e_std_meta.txt', test_name + '_e_std', [0, 0.4], N=10000, period=11)
    plots(res_dir, 'e_mean_meta.txt', test_name + '_e_mean', [-0.01, 0.01], N=10000, period=11)
