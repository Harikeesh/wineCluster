
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap

# Title of the app
st.title('Wine Quality Clustering K-Means, Hierarchical, DBSCAN, GMM)')

# Load dataset from URL
@st.cache
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    return data

data = load_data()

# Display the data
st.write("### Wine Quality Dataset")
st.dataframe(data)

# Basic information about the dataset
st.write("### Dataset Information")
st.write(data.info())

# Summary statistics
st.write("### Summary Statistics")
st.write(data.describe())

# Convert quality column to categorical
data['quality'] = data['quality'].astype('category')

# Sidebar options for clustering algorithm selection
st.sidebar.header("Clustering Options")
algorithm = st.sidebar.selectbox('Select Clustering Algorithm', 
                                 ('K-Means', 'Hierarchical (Agglomerative)', 'DBSCAN', 'Gaussian Mixture (GMM)'))

# Sidebar for additional parameters
if algorithm == 'K-Means':
    clusters = st.sidebar.slider('Number of clusters (K-Means)', 2, 10, 3)
elif algorithm == 'Hierarchical (Agglomerative)':
    clusters = st.sidebar.slider('Number of clusters (Hierarchical)', 2, 10, 3)
elif algorithm == 'DBSCAN':
    eps = st.sidebar.slider('EPS (DBSCAN)', 0.1, 1.0, 0.5)
    min_samples = st.sidebar.slider('Min Samples (DBSCAN)', 1, 10, 5)
else:
    clusters = st.sidebar.slider('Number of components (GMM)', 2, 10, 3)

# Standardizing the data before clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(['quality'], axis=1))

# Apply clustering based on user choice
if algorithm == 'K-Means':
    model = KMeans(n_clusters=clusters)
elif algorithm == 'Hierarchical (Agglomerative)':
    model = AgglomerativeClustering(n_clusters=clusters)
elif algorithm == 'DBSCAN':
    model = DBSCAN(eps=eps, min_samples=min_samples)
else:
    model = GaussianMixture(n_components=clusters)

# Fit the model
model.fit(data_scaled)

# If the algorithm has labels (or for GMM, predict), add them to the data
if algorithm == 'Gaussian Mixture (GMM)':
    data['Cluster'] = model.predict(data_scaled)
else:
    if hasattr(model, 'labels_'):
        data['Cluster'] = model.labels_

# Display the clusters
st.write(f"### Clustered Data using {algorithm}")
st.write(data.head())

# PCA for 2D Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)

# Plotting the PCA clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis')
plt.title(f'PCA of Wine Clusters ({algorithm})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
st.pyplot(plt)

# UMAP for 2D Visualization
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_data = umap_model.fit_transform(data_scaled)

# Plotting the UMAP clusters
plt.figure(figsize=(10, 6))
plt.scatter(umap_data[:, 0], umap_data[:, 1], c=data['Cluster'], cmap='plasma')
plt.title(f'UMAP of Wine Clusters ({algorithm})')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
st.pyplot(plt)

# Evaluation metrics section
st.write("### Clustering Evaluation Metrics")

if len(set(data['Cluster'])) > 1:  # Ensure we have more than one cluster
    # Silhouette Score
    silhouette = silhouette_score(data_scaled, data['Cluster'])
    st.write(f"Silhouette Score: {silhouette}")
    
    # Calinski-Harabasz Index
    calinski_harabasz = calinski_harabasz_score(data_scaled, data['Cluster'])
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
    
    # Davies-Bouldin Score
    davies_bouldin = davies_bouldin_score(data_scaled, data['Cluster'])
    st.write(f"Davies-Bouldin Score: {davies_bouldin}")
else:
    st.write("Not enough clusters to calculate evaluation metrics.")

# Sidebar option for additional visualizations
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
