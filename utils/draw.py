from matplotlib import pyplot as plt
import time
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
import pandas as pd
import numpy as np


def plot_embedding(X1, X2, y):
    time_start = time.time()
    df1 = pd.DataFrame(X1)
    df2 = pd.DataFrame(X2)

    # df_y = pd.DataFrame(y)
    # df['label'] = df['y'].apply(lambda i: str(i))
    # df_subset = df.loc[rndperm[:N],:].copy()
    # data_subset = df_subset[feat_cols].values

    df_subset1 = df1.copy()
    df_subset2 = df2.copy()
    # df_subset_y = df_y.copy()

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300)
    tsne_results1 = tsne.fit_transform(df1.values)
    tsne_results2 = tsne.fit_transform(df2.values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset1['tsne-2d-one'] = tsne_results1[:, 0]
    df_subset1['tsne-2d-two'] = tsne_results1[:, 1]
    df_subset2['tsne-2d-three'] = tsne_results2[:, 0]
    df_subset2['tsne-2d-four'] = tsne_results2[:, 1]

    # df_subset1['y'] = df_y.values
    # df_subset2['y'] = df_y.values

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    # ax1.scatter(df_subset['tsne-2d-one'], df_subset['tsne-2d-two'])
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # hue="y",
        palette=sns.color_palette("hls", 50),
        data=df_subset1,
        legend="full",
        alpha=1.0,
        ax=ax1
    )

    ax2 = plt.subplot(1, 2, 2)
    # ax2.scatter(df_subset2['tsne-2d-three'], df_subset2['tsne-2d-four'])
    sns.scatterplot(
        x="tsne-2d-three", y="tsne-2d-four",
        # hue="y",
        palette=sns.color_palette("hls", 50),
        data=df_subset2,
        legend="full",
        alpha=1.0,
        ax=ax2
    )

    plt.savefig('test.png', dpi=500, bbox_inches='tight')
    plt.show()

    print("Time: ", time.time() - time_start)


# def drawTSNE(vector, labels):

if __name__ == '__main__':
    X = np.random.randn(100, 64)
    y = np.array([([np.random.randint(0, 2) for i in range(10)]) for j in range(100)])
    plot_embedding(X, X, y)

