#%% md
# # # Homework 2: Unsupervised representation learning
#%% md
# # Download data, install packages
#%%
# !wget https://github.com/jertubiana/jertubiana.github.io/raw/master/misc/MLCB_2023_HW2_Data.pkl.zip
# !unzip MLCB_2023_HW2_Data.pkl.zip
# !pip install "ipyvolume>=0.6.2" "ipyvue>=1.9.1"
# !pip install plotly
#%% md
# # ## Load packages, load data
#%%
import math

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

path = 'MLCB_2023_HW2_Data.pkl'
env = pickle.load(open(path, 'rb'))
# Spikes, Time, Coordinates, Regions, Region_names = env['Spikes'], env['Time'], env['Coordinates'], env['Regions'], env[
#     'Region names']

Region_names = env['Region names']
#%% md
# # ## Visualize a short section of the recording as raster plot
#%%
fig, ax = plt.subplots(figsize=(10, 10))
sns.set_style("ticks")
ax.matshow(1 - env['Spikes'].T, aspect='auto')
plt.xticks(np.arange(len(env['Time']))[::1000], ['%.f' % t for t in env['Time'][::1000]])
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Neuron', fontsize=20)
plt.title('Spike raster plot', fontsize=20)
#%% md
# # ## Visualize the coordinates the neurons, colored by the region labels.
#%%
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(env['Coordinates'][:, 0], env['Coordinates'][:, 1], c=env['Regions'], s=10, alpha=0.1)
#%% md
# # ## A 3D-visualization of the recording with ipyvolume
# # This section does not run on Google Colab, but you can try it out on your local Python
#%%
import plotly.graph_objects as go
import numpy as np
#%%
# Create a scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=env['Coordinates'][:, 0],
    y=env['Coordinates'][:, 1],
    z=env['Coordinates'][:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=env['Regions'],
        opacity=0.1
    )
)])

# Set labels
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    title='3D Visualization of Neurons'
)

# Show the plot
fig.show()
#%%

#%% md
# # ## Subsampling the dataset to speed-up the analysis
#%%
np.random.seed(0)
active_neurons = (env['Spikes'].mean(0) != 0)  # Remove all inactive neurons
random_subset = active_neurons & (np.random.rand(env['Spikes'].shape[-1]) < 0.1)  # Keep only 10% of the neurons.
Spikes = env['Spikes'][:, random_subset].T
Coordinates = env['Coordinates'][random_subset]
Regions = env['Regions'][random_subset]

#%% md
# # Part II: Dataset Analysis (Representation learning) 
#%% md
# ## Background:
# Progress in functional brain imaging techniques enable the simultaneous recording of the activity of thousands of neurons simultaneously. 
# 
# An open question is how to relate the observed spontaneous activity to the underlying 
# 1) neuronal assemblies (sets of neurons involved in a specific task, e.g. sensory, motor or cognitive) 
# 1) connectome (set of synapses). 
# 
# Herein, we focus on a recording of spontaneous brain activity of a larval zebrafish. Activity was measured using genetically modified fish expressing a fluorescent calcium reporter and with light-sheet fluorescence microscopy.
#  
# This dataset was collected in the laboratory of Pr. Georges Debregeas (Reference: [elife](https://elifesciences.org/articles/83139)), and consists of:
# 
# 1. A matrix $X_{ij}$ of binary spikes of format [Time x Neurons].
# 2. 3D coordinates of the neurons [Neurons x 3].
# 3. An integer brain region label for each neuron, assigned based on the morphology $ M_i $ [Neurons,].
# 
# Our goal is to recover, in an unsupervised manner, the morphological regions from the correlation structure of the data. 
# A Google Colab Notebook with package installation, dataset download and visualizations is available from: [Colab Notebook](https://colab.research.google.com/drive/1qSQelSprDyMKF_uRQ6ihDZWWnfdF7nOB?usp=sharing)
#%%
print(f"Spikes shape: {Spikes.shape}")
print(f"Coordinates shape: {Coordinates.shape}")
print(f"Regions shape: {Regions.shape}")
#%%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from plotly.subplots import make_subplots


#%% md
#  
# ### Low-dimensional embeddings: PCA vs t-SNE
#  
# We define a custom distance metric between two neurons as one minus the absolute value of the Pearson correlation coefficient:
# $$
# \[ D_{ij} = 1 - \left| \frac{ \text{Cov}(X_i, X_j) }{ \sqrt{ \text{Var}(X_i) \cdot \text{Var}(X_j) } } \right| \]
# $$
# 
# 1. Using the scikit-learn default implementations, calculate two-dimensional embeddings $ R_{PCA/t-SNE} $ of the data using 1) PCA and 2) t-SNE with a perplexity of 5, 10, 20, 50, 100. Visualize them using scatter plots, where the points are colored by the brain region.
# 
#%%
def custom_distance(X):
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(X)
    # Calculate the distance matrix based on the given formula
    distance_matrix = 1 - np.abs(corr_matrix)
    return fix_distance_matrix(distance_matrix)


def fix_distance_matrix(distance_matrix):
    # For numerical reasons, ensure the correlation matrix satisfies the properties of a valid correlation
    sym_corr_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure symmetry
    np.fill_diagonal(sym_corr_matrix, 0)  # Ensure the diagonal is zero
    return sym_corr_matrix

#%%
def calculate_pca_embeddings(Spikes, with_scaler=False):
    pca = PCA(n_components=2)
    if with_scaler:
        Spikes = StandardScaler().fit_transform(Spikes)
    pca_result = pca.fit_transform(Spikes)

    return pca, pca_result


def calculate_tsne_embeddings(distances, perplexities):
    tsne_results = {}
    for perplexity in perplexities:
        tsne = TSNE(n_components=2, perplexity=perplexity, metric='precomputed', init='random')
        tsne_results[perplexity] = tsne.fit_transform(distances)

    return tsne_results

#%%
def plot_embeddings(
        pca_result, tsne_results, regions, region_names, perplexities,
        n_top_regions_for_legend=10,
        figsize=(15, 5), alpha_pca=0.5, alpha_tsne=0.1
):
    num_rows = (len(perplexities) + 1) // 2 + 1
    fig, axes = plt.subplots(num_rows, 2, figsize=(figsize[0], figsize[1] * num_rows))

    top_regions = [
        region_names[region_id]
        for region_id
        in pd.Series(Regions).value_counts().index[:n_top_regions_for_legend]
    ]

    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='C' + str(i), label=region_name
        )
        for i, region_name in enumerate(top_regions)
    ]

    def plot_legend(ax):
        ax.legend(
            handles, top_regions,
            loc='center', fontsize='x-large', markerscale=2)
        ax.axis('off')

    plot_legend(axes[0, 0])

    axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], c=regions, alpha=alpha_pca)
    axes[0, 1].set_title('PCA')

    for i, perplexity in enumerate(perplexities):
        row = i // 2 + 1
        col = i % 2
        axes[row, col].scatter(tsne_results[perplexity][:, 0], tsne_results[perplexity][:, 1], c=regions,
                               alpha=alpha_tsne)
        axes[row, col].set_title(f't-SNE (perplexity={perplexity})')

    if len(perplexities) % 2 == 1:
        row = (len(perplexities) + 1) // 2
        plot_legend(axes[row, 1])

    plt.tight_layout()
    plt.show()
#%%
distance_matrix = custom_distance(Spikes)
#%%
pca, pca_embeddings = calculate_pca_embeddings(Spikes)
#%%
perplexities = [5, 10, 20, 50, 100]
tsne_embeddings = calculate_tsne_embeddings(distance_matrix, perplexities)
#%%
plot_embeddings(pca_embeddings, tsne_embeddings, Regions, Region_names, perplexities)
#%%

#%% md
# #### Which method qualitatively separates the regions in the most convincing way?
#%% md
# Answer: t-SNE with perplexity 50 separates the regions in the most convincing way. The regions are more clearly separated compared to PCA and other t-SNE perplexities.
#%% md
# # 2. **Preserving global data structure.** 
# For each method, calculate the spearman correlation coefficient between the original dissimilarity matrix $ D_{ij} $ and the Euclidean distance matrix in the embedding: $ D_{ij}^{R} = || R_i - R_j || $. 
# # 
#%%
def calculate_spearman_correlation(distance_matrix, pca_distance_matrix, tsne_distance_matrices):
    def calculate_spearman(original_dist, embedding_dist):
        original_dist = fix_distance_matrix(original_dist)
        embedding_dist = fix_distance_matrix(embedding_dist)
        return spearmanr(squareform(original_dist), squareform(embedding_dist)).correlation

    pca_spearman = calculate_spearman(distance_matrix, pca_distance_matrix)
    tsne_spearmans = {perplexity: calculate_spearman(distance_matrix, tsne_distance_matrix) for
                      perplexity, tsne_distance_matrix in tsne_distance_matrices.items()}

    return pca_spearman, tsne_spearmans

#%%
# Euclidean distance matrix for PCA
pca_distance_matrix = pairwise_distances(pca_embeddings)
# Euclidean distance matrices for t-SNE
tsne_distance_matrices = {
    perplexity: pairwise_distances(tsne_result)
    for perplexity, tsne_result in
    tsne_embeddings.items()
}

#%%
# Spearman correlation for PCA
pca_spearman, tsne_spearmans = calculate_spearman_correlation(distance_matrix, pca_distance_matrix,
                                                              tsne_distance_matrices)

print("Spearman correlation for PCA:", pca_spearman)
for perplexity, spearman_corr in tsne_spearmans.items():
    print(f"Spearman correlation for t-SNE (perplexity={perplexity}):", spearman_corr)

#%% md
#  
# ##  3. **Preserving local data structure.** 
# We define the K-nearest neighbors (K-NN) graph $ A_{ij} $ as:
#  
#  $$
#  \[ A_{ij} = 
#  \begin{cases} 
#  1 & \text{if } j \text{ is one of the K – nearest neighbors of } i, \text{ or vice versa} \\ 
#  0 & \text{otherwise} 
#  \end{cases}
#  \]
#  $$
#  
#  
#%% md
# a) With K=50, calculate the K-NN graphs using the original $ D_{ij} $ and, for each method, $ D_{ij}^{R} $. They are denoted by $ A $ and $ A^{R} $, respectively.
#%%
def knn_graph(data, k):
    knn = NearestNeighbors(n_neighbors=k).fit(data)
    knn_matrix = knn.kneighbors_graph(data).toarray()
    # force symmetric metric
    return np.maximum(knn_matrix, knn_matrix.T)


K = 50

# Original K-NN graph
original_knn_graph = knn_graph(distance_matrix, k=K)

# K-NN graph for PCA
pca_knn_graph = knn_graph(pca_distance_matrix, k=K)

# K-NN graphs for t-SNE
tsne_knn_graphs = {perplexity: knn_graph(tsne_distance_matrix, k=K) for perplexity, tsne_distance_matrix in
                   tsne_distance_matrices.items()}

#%% md
# b) Calculate the fraction of edges in the original graph $ A $ that are preserved in $ A^{R} $. 
#%%
def fraction_of_preserved_edges(original_graph, embedded_graph):
    preserved_edges = np.sum((original_graph == 1) & (embedded_graph == 1))
    total_edges = np.sum(original_graph == 1)
    return preserved_edges / total_edges

#%%
pca_preservation = fraction_of_preserved_edges(original_knn_graph, pca_knn_graph)
tsne_preservations = {
    perplexity: fraction_of_preserved_edges(original_knn_graph, tsne_knn_graph)
    for perplexity, tsne_knn_graph in tsne_knn_graphs.items()
}

print(f"Fraction of preserved edges for PCA: {pca_preservation:.4f}")
for perplexity, preservation in tsne_preservations.items():
    print(f"Fraction of preserved edges for t-SNE (perplexity={perplexity}): {preservation:.4f}")
#%% md
# ### Conclude on which method best preserves the local neighborhoods?
# 
# #### Answer: 
# t-SNE with a perplexity of 100 best preserves the local neighborhoods
#%% md
# ## 4. **Preserving morphology.** 
# Using the same graphs $ A $ and $ A^{R} $, calculate the fraction of pairs of neighbors that belong to the same region:
# $$
# \[ f = \frac{\sum_{i,j} A_{i,j} \delta_{M_i, M_j}}{\sum_{i,j} A_{i,j}} \]
# $$
#%%
def fraction_of_same_region_neighbors(knn_graph, regions):
    same_region_neighbors = np.sum(
        (knn_graph == 1) & (np.equal.outer(regions, regions))
    )
    total_neighbors = np.sum(knn_graph == 1)
    return same_region_neighbors / total_neighbors


#%%
original_knn_graph_region_preservation = fraction_of_same_region_neighbors(original_knn_graph, Regions)
pca_region_preservation = fraction_of_same_region_neighbors(pca_knn_graph, Regions)
tsne_region_preservations = {
    perplexity: fraction_of_same_region_neighbors(tsne_knn_graph, Regions)
    for perplexity, tsne_knn_graph in tsne_knn_graphs.items()
}
#%%
print(f"Fraction of same region neighbors for Original: {original_knn_graph_region_preservation:.4f}")
print(f"Fraction of same region neighbors for PCA: {pca_region_preservation:.4f}")
for perplexity, preservation in tsne_region_preservations.items():
    print(f"Fraction of same region neighbors for t-SNE (perplexity={perplexity}): {preservation:.4f}")

#%% md
# ### Which method best preserves morphological similarity?
# #### Answer: 
# t-SNE with a perplexity of 100 best preserves morphological similarity
#%% md
# 
# ## B. Feature extraction: PCA vs NMF.
# 1. For each of the first two principal components, visualize the corresponding principal loadings (projection matrix) in space, using scatter plots and/or the ipyvolume visualization provided. 
#%%
def plot_loadings(coordinates, loadings, title, n_loadings_to_show=2):
    # Calculate the number of rows needed
    num_cols = 2
    num_rows = int(math.ceil((n_loadings_to_show) / num_cols))

    # Create a subplot with the required number of rows and columns
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=[f'Loading {i + 1}' for i in range(n_loadings_to_show)],
        specs=[[{'type': 'scatter3d'} for _ in range(num_cols)] for _ in range(num_rows)]
    )



    # Add a scatter plot for each loading
    for i in range(n_loadings_to_show):
        loading = loadings[i, :]
        normalized_loadings = loading + np.abs(np.min(loading))
        normalized_loadings /= np.max(normalized_loadings)

        row = (i // num_cols) + 1
        col = (i % num_cols) + 1
        scatter = go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=normalized_loadings,
                opacity=1
            ),
        )
        fig.add_trace(scatter, row=row, col=col)

        # Set the layout
        fig.update_layout(
            height=400 * num_rows,  # Adjust height as needed
            title_text=title,
            showlegend=False,
        )

    # Show the plot
    fig.show()

#%%
pca_T, pca_embeddings_T = calculate_pca_embeddings(Spikes.T)
plot_loadings(Coordinates, pca_T.components_, 'PCA Loadings')
#%%
pca_T, pca_embeddings_T = calculate_pca_embeddings(Spikes.T)
plot_loadings(Coordinates, pca_T.components_, 'PCA Loadings')
#%%
def plot_NMF_loadings2(n_components, Spikes, ):
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    _ = nmf.fit_transform(Spikes.T)
    title = f'NMF Loadings with {n_components} components'
    plot_loadings(Coordinates, pca_T.components_, title, n_loadings_to_show=2, )


#%%
plot_NMF_loadings2(2, Spikes)
#%% md
# Do they localize onto some of the morphological regions?
#%% md
# #### Answer:
# In out eyes, both graphs did not provide clear localization onto the morphological regions
#%% md
# # 2. Using the scikit-learn’s implementation, apply Non-Negative Matrix Factorization with n_components=20 and visualize two components in space.
#%%
plot_NMF_loadings2(20, Spikes)
#%% md
#  ### Do they localize onto some of the morphological regions?
#%% md
# #### Answer:
# 
#%% md
# # 3. Repeat for n_components = 2, 5 and 50.
# # 
#%%
plot_NMF_loadings2(50, Spikes)
#%% md
# ### How do the components change?
#%% md
# #### Answer: