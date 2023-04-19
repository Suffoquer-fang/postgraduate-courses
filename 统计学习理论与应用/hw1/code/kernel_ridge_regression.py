from math import exp
import numpy as np
import random
import matplotlib.pyplot as plt


Y_FUNCTION = lambda x: -5 + 20 * x - 16 * x * x

def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)

def generate_dataset(N=30):
    X = np.random.uniform(0, 1, N)
    eps = np.random.standard_normal(N)
    
    Y = Y_FUNCTION(X)
    Y = Y + 0.4 * eps

    return (X, Y)

class KernelRidgeRegression:
    def __init__(self, kernel_fn, lamb):
        self.kernel_fn = kernel_fn
        self.lamb = lamb

    def fit(self, X, Y):
        self.X = X 
        self.Y = Y
        N = Y.shape[0]

        gram_matrix = np.zeros((N, N))

        # self.lamb = max(self.lamb, 1e-7)
        for i in range(N):
            for j in range(N):
                gram_matrix[i, j] = self.kernel_fn(X[i], X[j])
        
        gram_matrix = np.matrix(gram_matrix)
        gram_matrix += self.lamb * np.identity(N)
    
        self.reverse_matrix = gram_matrix.I
        self.predict_matrix = np.matmul(self.Y, self.reverse_matrix)

    def predict(self, x):
        kernel_vec = [self.kernel_fn(x_i, x) for x_i in self.X]
        kernel_vec = np.matrix(kernel_vec)

        y_hat = np.matmul(self.predict_matrix, kernel_vec.transpose()) 
        return y_hat[0, 0]


def plot_dataset(X, Y, **kwargs):
    plt.scatter(X, Y, **kwargs)

def plot_curve(curve_fn, **kwargs):
    X_curve = np.arange(0, 1, 0.01)
    Y_curve = [curve_fn(x) for x in X_curve]
    plt.plot(X_curve, Y_curve, **kwargs)

def kernel_fn(x, y):
    # return (1 + x*y) ** 9
    return min(x, y)




if __name__ == "__main__":
    set_seed(42)
    X, Y = generate_dataset()
    
    

    plt.figure(figsize=(10, 8))
    plot_dataset(X, Y, label='dataset', c='green', s=20)
    plot_curve(Y_FUNCTION, label='target fn')
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title('Prediction Function With Different Kernels', fontsize=15)
    
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)  
    
    # for lamb in [0.001, 0.01, 0.1, 1.0]:
    #     model = KernelRidgeRegression(lambda x, y: exp(-1*(x-y)**2), lamb)
    #     model.fit(X, Y)
    #     plot_curve(model.predict, label=f'predict fn (lambda = {lamb})', linestyle='-.')
    

    model = KernelRidgeRegression(lambda x, y: min(x, y), 0.1)
    model.fit(X, Y)
    plot_curve(model.predict, label=f'predict fn (min(x, y))', linestyle='-.')

    
    model = KernelRidgeRegression(lambda x, y: (1+x*y)**1, 0.1)
    model.fit(X, Y)
    plot_curve(model.predict, label=f'predict fn (1+xy)', linestyle='-.')

    model = KernelRidgeRegression(lambda x, y: (1+x*y)**2, 0.1)
    model.fit(X, Y)
    plot_curve(model.predict, label=f'predict fn ((1+xy)^2)', linestyle='-.')

    model = KernelRidgeRegression(lambda x, y: (1+x*y)**9, 0.1)
    model.fit(X, Y)
    plot_curve(model.predict, label=f'predict fn ((1+xy)^9)', linestyle='-.')

    model = KernelRidgeRegression(lambda x, y: exp(-1*(x-y)**2), 0.1)
    model.fit(X, Y)
    plot_curve(model.predict, label=f'predict fn (exp(-(x-y)^2))', linestyle='-.')

    plt.legend(fontsize=15)

    # plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95)
    
    plt.show()
    # plt.savefig('./exp(-(x-y)^2).png')
