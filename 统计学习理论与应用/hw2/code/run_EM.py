import numpy as np
from tqdm import tqdm 
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_top_words(model, words_list, weights_list, title):
    fig, axes = plt.subplots(10, 5, figsize=(26, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(words_list):
        # top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = topic
        weights = weights_list[topic_idx]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 25})
        ax.invert_yaxis()
        ax.tick_params(axis="y", which="major", labelsize=25)
        ax.tick_params(axis="x", which="major", labelsize=20)
        
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.88, bottom=0.05, wspace=0.90, hspace=0.3)
    # plt.show()
    plt.savefig('./topic-50.pdf')




def load_dataset():
    wordid2word = {}
    with open('./dataset/20news.vocab', 'r') as f:
        # W = len(f.readlines())
        lines = f.readlines()
    
    W = len(lines)

    for line in lines:
        wordid, word, _ = line.strip().split('\t')
        wordid = int(wordid)
        wordid2word[wordid] = word 
    

    with open('./dataset/20news.libsvm', 'r') as f:
        lines = f.readlines()

    D = len(lines)
    T = np.zeros((D, W), dtype=int)
    for line in tqdm(lines):
        docid, word_list = line.strip().split('\t')
        docid = int(docid)
        word_list = word_list.split(' ')
        for wc in word_list:
            wordid, count = wc.split(':')
            wordid = int(wordid)
            count = int(count)
            T[docid, wordid] = count
    return T, wordid2word  

class EM:
    def __init__(self, T, K) -> None:
        self.T = T 
        self.K = K
        self.D, self.W = T.shape
        self.mu = np.random.random((self.W, K))
        self.pi = np.random.random(K)

        self.nd = np.sum(T, axis=1)
        self.mu = self.mu / np.sum(self.mu, axis=0, keepdims=True)
        self.r = np.zeros((self.D, self.K))

    def calculate_r(self):
        mu = self.mu + 1e-30
        log_mu = np.log(mu, dtype=np.float64)
        log_p_dk = np.dot(self.T, log_mu)

        max_log = np.max(log_p_dk, axis=1, keepdims=True)
        log_p_dk = log_p_dk - max_log
        p_dk = np.exp(log_p_dk)
        for d in tqdm(range(self.D)):
            prod_k = np.zeros(self.K, dtype=np.float64)
            for k in range(self.K):
                prod_k[k] = p_dk[d, k]
            prod_k = np.multiply(self.pi, prod_k)
            for k in range(self.K):
                self.r[d, k] = prod_k[k] / np.sum(prod_k)

    def calculate_r_naive(self):
        
        for d in tqdm(range(self.D)):
            prod_k = np.zeros(self.K)
            for k in range(self.K):
                prod = 1.0
                pos_w = np.where(self.T[d] > 0)
                # print(pos_w)
                for w in pos_w[0]:
                    prod *= (self.mu[w, k] ** self.T[d, w])  
                prod_k[k] = prod 

            prod_k = np.multiply(self.pi, prod_k)
            self.r[d, :] = prod_k[:] / (np.sum(prod_k) + 0.0 )
        
    def do_iteration(self):
        print('Begin Iteration')
        self.calculate_r()
        self.new_pi = np.zeros_like(self.pi)
        self.new_mu = np.zeros_like(self.mu)
        for k in tqdm(range(self.K)):
            top, bottom = 0.0, 0.0
            for d in range(self.D):
                top += self.r[d, k]

                for k_ in range(self.K):
                    bottom += self.r[d, k_]
            self.new_pi[k] = top / bottom
        
        bottom_k = np.dot(self.nd, self.r)
        top_wk = np.dot(self.T.transpose(), self.r)
        for k in tqdm(range(self.K)):
            self.new_mu[:, k] = top_wk[:, k] / (bottom_k[k] + 0.0 )
        
        delta = np.sum((self.mu - self.new_mu) ** 2)
        self.mu = self.new_mu.copy()
        self.pi = self.new_pi.copy()

        print('Done Iteration With Delta =', delta)
        print()
        if delta < 1e-6:
            return True
        return False


    def most_frequent_words(self, k, topk=10):
        mu = self.mu[:, k]
        mu_idx = np.argsort(mu)
        # print(mu)
        w_list = mu_idx[-topk:]
        weights = mu[w_list]
        return w_list.tolist()[::-1], weights.tolist()[::-1]


    def calculate_dbi(self):    
        C = np.argmax(self.r, axis=1)
        S = np.zeros(self.K)
        centers = np.zeros((self.K, self.W))

        for k in range(self.K):
            d_set = np.where(C == k)[0]
            vectors = self.T[d_set, :]
            
            len_vectors = self.nd[d_set]

            vectors = vectors / len_vectors[:, None]
            center = np.mean(vectors, axis=0)
            dist = np.dot(center, vectors.transpose())
            S[k] = np.mean(dist)
            centers[k] = center 

        dbi = 0.0
        for i in range(self.K):
            max_temp = 0.0
            for j in range(self.K):
                if j == i: continue
                temp = S[i] + S[j]
                temp /= np.dot(centers[i], centers[j]) 
                max_temp = max(max_temp, temp)
            dbi += max_temp / self.K 

        return dbi



if __name__ == "__main__":
    np.random.seed(12345)
    K = 50
    L = 10
    T, word_dict = load_dataset()
    model = EM(T, K)

    done = False
    for _ in range(L):
        done = model.do_iteration()
        if done: break

    words_list = []
    weights_list = []
    for k in range(K):
        w_list, weights = model.most_frequent_words(k, topk=10)
        words = [word_dict[w] for w in w_list]
        print('k =', k)
        print('words =', words)
        print('weights =', weights)
        print()
        words_list.append(words[:3])
        weights_list.append(weights[:3])

    plot_top_words(_, words_list, weights_list, f'Most Frequent Words in Topics (K = {K})')


    
