from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(X):
    tsne = TSNE(n_components=2, verbose=True)
    plt.figure()
    out = tsne.fit_transform(X[:, :].T)
    out1 = out[:100]
    out2 = out[100:]
    plt.scatter(out1[:, 0], out1[:, 1], label='top100', alpha=0.5, s=10)
    plt.scatter(out2[:, 0], out2[:, 1], label='other', alpha=0.5, s=10)

    plt.legend()
    # plt.show()
    plt.title('T-SNE Visualization of All Eigen Vectors')
    plt.savefig('./all_eigen.pdf', dpi=300)



data = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

data = [data[i][0].reshape(-1).numpy() for i in range(len(data))]

np.random.seed(12345)


N = len(data)
X = np.array(data)
X_mean = np.mean(X, axis=0)
# X_mean = np.zeros_like(X)
X = X - X_mean 


X = X.transpose()

S = 1 / N * np.matmul(X, X.T)
eig_vals, eig_vectors = np.linalg.eig(S)

eig_val_index = np.argsort(eig_vals)  
eig_vals = eig_vals[eig_val_index]

R = eig_vals / np.sum(eig_vals)
print(R[-10:])


top100_vecs = eig_vectors[:, eig_val_index]
top100_vecs = top100_vecs[:, ::-1]

print(top100_vecs.shape)
print(top100_vecs[0])

visualize(top100_vecs)

plt.figure(figsize=(20, 5))
font = {
    'weight' : 'normal',
    'size' : 20,
}
plt.title('Resulting Images with Different Information Preserved')
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2)
for i, preserve in enumerate([0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 'original']):
    t = 0.0
    idx = -1

    if preserve == 'original':
        
        plt.subplot(2, 8, i+1)
    
        plt.imshow(data[5].reshape(28, 28), cmap=plt.cm.binary)
        plt.axis("off")
        plt.title(f'{preserve}', font)

        plt.subplot(2, 8, i+1+8)
        plt.axis("off")
        plt.imshow(data[800].reshape(28, 28), cmap=plt.cm.binary)
        continue

    while t < preserve and idx > -len(R):
        t += R[idx]
        idx -= 1

    index = eig_val_index[idx+1: ] 

    U1 = eig_vectors[:, index] 

    print(U1.shape)

    Y = np.matmul(U1.T, X)
    X_hat = np.matmul(U1, Y)

    X_hat = X_hat.transpose()
    X_hat = X_hat + X_mean

    
    plt.subplot(2, 8, i+1)
    
    plt.imshow(X_hat[5].reshape(28, 28), cmap=plt.cm.binary)
    plt.axis("off")
    plt.title(f'{100*preserve}%', font)

    plt.subplot(2, 8, i+1+8)
    plt.axis("off")
    plt.imshow(X_hat[800].reshape(28, 28), cmap=plt.cm.binary)
# plt.show()
plt.savefig('./different_information.pdf', dpi=300)
