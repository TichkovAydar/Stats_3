import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



def initial_list():
    A = np.array(list([2, 4, 6, 8, 10]))
    B = np.array(list([3.5, 6.0, 7.0, 6.0, 7.5]))
    return A, B


def stat_array(A, B):
    # Выводим информацию о массиве
    print("Suppose we have to 2 arrays of random values \nA ---> {}\nB ---> {}".format(A, B))
    print("Выборочная ковариация равняется ---> {0:.3f}".format(correl(A, B)))
    print("Выборочная дисперсия для каждого массива равны \ns(A) ---> {0:.3f}\ns(B) ---> {1:.3f}".format(disp(A),
                                                                                                         disp(B)))
    print("Выборочная корреляции равняется ---> {0:.3f}".format(covar(A, B)))
    print("Коэффициент регрессии B на A равняется ---> {0:.3f}".format(line(A, B)))
    print("Уравнение прямой регрессии B на A равняется ---> y = {0:.3f} * x + {1:.3f}".format(line(A, B), (
            B.mean() - line(A, B) * A.mean())))
    print("Коэффициент регрессии A на B равняется ---> {0:.3f}".format(line(B, A)))
    print("Уравнение прямой регрессии A на B равняется ---> x = {0:.3f} * y + {1:.3f}".format(line(B, A), (
            B.mean() - line(B, A) * A.mean())))

    def graph(A, B, b1, b0, x_range=(-A[-1], A[-1]),
              label="Линия регрессии"):
        plt.scatter(A, B)
        plt.plot([x_range[0], x_range[1]],
                 [x_range[0] * b1 + b0, x_range[1] * b1 + b0],
                 c="g", linewidth=2, label=label)
        plt.legend()
        plt.show()
    x_range = (0.8*A.min(), 1.2*A.max())
    graph(A, B, line(A, B),  B.mean() - line(A, B) * A.mean(), x_range=x_range)


def read_csv_file(filename):
    csv = pd.read_csv(filename, sep=";")
    A = np.array(csv["e"])
    B = np.array(csv["n"])
    return A, B


def read_correl_file(filename):
    csv_correl = np.array(pd.read_csv(filename, sep=";"))
    A = [0.5, 0.6, 0.7, 0.8, 0.9]
    B = [0.5, 0.6, 0.7, 0.8]
    C = []
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(csv_correl[i][j]):
                C.append([A[i], B[j]])
    D = np.fastCopyAndTranspose(C)
    return D[0], D[1]


def covar(A, B):
    return np.corrcoef(B, A)[0][1]


def disp(D):
    C = 0
    for i in range(len(D)):
        C = C + (D[i] * D[i] - (np.mean(D)) ** 2)
    return math.sqrt(C / len(D))


def correl(A, B):
    return np.cov(A, B, bias=True)[0][1]


def line(A, B):
    l = covar(A, B) * disp(B) / disp(A)
    return l
