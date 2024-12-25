import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import skfuzzy as fuzz
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

from config import AnalysisConfig

seed = 7
np.random.seed(seed)
torch.manual_seed(seed)

def clean(data, replace = True, remove_success = True):
    exclude_cols1 = ["founder_uuid", "name", "org_name", "persona"]

    exclude_cols2 = ["success"]
    
    data = data.drop(columns=exclude_cols1)

    if remove_success:
        data = data.drop(columns=exclude_cols2)

    # Define a mapping for the acquisition ranges
    acquisition_mapping = {
        'nope': 0,
        'undisclosed': 1,
        'l20': 2,
        '20_50': 3,
        '50_150': 4,
        '150_500': 5,
        'g500': 6,
    }

    # Apply the mapping while leaving NaN values as is
    data['acquisition_experience'] = data['acquisition_experience'].map(acquisition_mapping)

    # Set 'yoe' (years of experience) to NaN if it is less than 0
    data.loc[data['yoe'] < 0, 'yoe'] = None

    # Set the specified columns to 0 if they are currently null
    columns_to_fill = ['founder_experience', 'acquisition_experience', 'acquirer_bigtech', 'ipo_experience']
    data[columns_to_fill] = data[columns_to_fill].fillna(0)

    # Binary columns: Convert True/False to 1/0 while keeping NaN
    binary_columns = ['ipo_experience', 'acquirer_bigtech']
    data[binary_columns] = data[binary_columns].replace({True: 1, False: 0})

    # Categorical column: Map values to integers while keeping NaN
    experience_mapping = {'nope': 0, 'unit': 1, 'multi': 2}
    data['founder_experience'] = data['founder_experience'].map(experience_mapping)

    if replace:
        columns_to_fill_with_mean = ['yoe', 'founder_experience']

        for col in columns_to_fill_with_mean:
            if col in data.columns:
                mean_value = data[col].mean()
                data[col].fillna(mean_value, inplace=True)

    return data

def check_na(data):
    # Analyze missing values before dropping them
    missing_values = data.isnull().sum()
    missing_values_percentage = (data.isnull().sum() / len(data)) * 100

    # Combine into a summary table
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_values_percentage
    }).sort_values(by='Missing Values', ascending=False)

    print(missing_summary)

#APPLY SMOTE

#from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import train_test_split

#X = data_success.drop(columns=['success'])
#y = data_success['success']

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the classes in the training data
#smote = SMOTE(random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Display the resampled data
#resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['success'])], axis=1)

#resampled_data['success'].value_counts()

#data = resampled_data.drop(columns=['success'])


def encode(data, num_epochs=50, dim = 10):
    # Add index column to preserve the original order
    data = data.reset_index(drop=True)
    data['index'] = data.index

    # Scale only numerical columns (excluding 'index')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=['index']))

    # Convert data and indexes to tensors
    data_tensor = torch.FloatTensor(scaled_data)
    index_tensor = torch.LongTensor(data['index'].values)

    # Combine tensors into a dataset
    dataset = TensorDataset(data_tensor, index_tensor)

    # Create DataLoader with shuffle=True
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 2: Define VAE Model
    class VAE(nn.Module):
        def __init__(self, input_dim, latent_dim=10):
            super(VAE, self).__init__()
            self.latent_dim = latent_dim
            
            #self.encoder = nn.Sequential(
            #    nn.Linear(input_dim, 64),
            #    nn.BatchNorm1d(64),
            #    nn.LeakyReLU(),
            #     nn.Linear(64, 32),
            #     nn.BatchNorm1d(32),
            #     nn.LeakyReLU(),
            # )

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(32, latent_dim)
            self.fc_logvar = nn.Linear(32, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )

        def encode(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            return recon_x, mu, logvar

    # Step 3: Training Setup
    def loss_function(recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

    # Initialize VAE
    input_dim = data_tensor.shape[1]
    latent_dim = dim
    vae = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    # Step 4: Train VAE
    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in dataloader:
            x, _ = batch  # x is the data, _ is the index
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        #print(f"Epoch {epoch + 1}, Loss: {train_loss / len(dataset):.4f}")

    # Step 5: Extract Latent Representations with Index Tracking
    vae.eval()
    latent_data = []
    original_indexes = []

    with torch.no_grad():
        for batch in dataloader:
            x, idx = batch  # Retrieve data and indexes
            mu, _ = vae.encode(x)
            latent_data.append(mu)
            original_indexes.extend(idx)

    # Combine latent data and indexes
    latent_data = torch.cat(latent_data, dim=0).numpy()
    original_indexes = np.array(original_indexes)

    # Step 6: Sort Latent Representations Back to Original Order
    sorted_indices = np.argsort(original_indexes)
    latent_data = latent_data[sorted_indices]

    # Step 7: Decode Latent Data Back to Original Space
    #decoded_data = []
    #with torch.no_grad():
    #     for i in range(0, len(latent_data), batch_size):
    #         latent_batch = latent_data[i:i+batch_size]
    #         latent_tensor = torch.FloatTensor(latent_batch)
    #         recon_data = vae.decode(latent_tensor)
    #         decoded_data.append(recon_data)

    # decoded_data = torch.cat(decoded_data, dim=0).numpy()

    # Step 8: Map Back to Original Scale
    #original_data_reconstructed = scaler.inverse_transform(decoded_data)
    #original_data_df = pd.DataFrame(original_data_reconstructed, columns=data.columns[:-1])  # Exclude 'index'

    latent_data_df = pd.DataFrame(latent_data, columns=[f"latent_{i}" for i in range(latent_dim)])
    return latent_data_df, scaler, vae

def encode_new_data(new_data, scaler, vae):
    """
    Encode a single new data point using the trained VAE encoder.

    Parameters:
        new_data (pd.DataFrame): A single new data point in the same format as training data (as a DataFrame row).
        scaler (StandardScaler): The scaler used to normalize the original training data.
        vae (VAE): The trained VAE model.

    Returns:
        np.array: The latent representation of the new data point.
    """
    # Step 1: Preprocess the new data
    new_data_scaled = scaler.transform(new_data)  # Scale using the trained scaler

    # Convert the scaled data into a tensor
    new_data_tensor = torch.FloatTensor(new_data_scaled)

    # Step 2: Encode the new data
    vae.eval()  # Set the VAE to evaluation mode
    with torch.no_grad():
        mu, logvar = vae.encode(new_data_tensor)  # Get the latent distribution
        latent_representation = mu.numpy()  # Use the mean as the deterministic encoding

    return latent_representation

# Function to calculate success proportion range for each clustering method
def calculate_success_range(data_success, cluster_labels, method_name):
    merged_data = data_success.copy()
    merged_data[method_name] = cluster_labels
    success_proportion = merged_data.groupby(method_name)['success'].mean()
    return success_proportion.max() - success_proportion.min()

def perform_clustering_analysis(data, data_success, n, num_clusters=5, num_epochs=50):
    # Initialize results dictionary
    results = {
        'KMeans': 0,
        'Spectral': 0,
        'Hierarchical': 0,
        'GMM': 0,
        'KMedoids_Labels': 0,
        'FuzzyCMeans_Labels': 0,
        'Random_Labels': 0
    }

    for i in range(n):
        seed = i
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Encode and scale data
        # latent_data_df = encode(data, num_epochs=num_epochs)
        scaler = StandardScaler()
        latent_data_df = data.copy()  # Assuming `encode` returns a DataFrame similar to `data`
        latent_data_scaled = scaler.fit_transform(latent_data_df)

        # K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=i+1)
        latent_data_df['KMeans_Labels'] = kmeans.fit_predict(latent_data_scaled)
        results['KMeans'] += calculate_success_range(data_success, latent_data_df['KMeans_Labels'], 'KMeans_Labels')

        # Spectral Clustering
        distances = pairwise_distances(latent_data_scaled)
        sigma = np.median(distances)
        gamma = 1 / (2 * sigma**2)
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='rbf', gamma=gamma, random_state=42)
        latent_data_df['Spectral_Labels'] = spectral.fit_predict(latent_data_scaled)
        results['Spectral'] += calculate_success_range(data_success, latent_data_df['Spectral_Labels'], 'Spectral_Labels')

        # Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
        latent_data_df['Hierarchical_Labels'] = hierarchical.fit_predict(latent_data_scaled)
        results['Hierarchical'] += calculate_success_range(data_success, latent_data_df['Hierarchical_Labels'], 'Hierarchical_Labels')

        # Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=num_clusters, random_state=i+1)
        latent_data_df['GMM_Labels'] = gmm.fit_predict(latent_data_scaled)
        results['GMM'] += calculate_success_range(data_success, latent_data_df['GMM_Labels'], 'GMM_Labels')

        # K-Medoids
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=i+1)
        latent_data_df['KMedoids_Labels'] = kmedoids.fit_predict(latent_data_scaled)
        results['KMedoids_Labels'] += calculate_success_range(data_success, latent_data_df['KMedoids_Labels'], 'KMedoids_Labels')

        # Fuzzy C-Means
        data_T = latent_data_scaled.T  # Transpose for skfuzzy compatibility
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data_T, c=num_clusters, m=2, error=0.005, maxiter=1000, init=None)
        cluster_labels = np.argmax(u, axis=0)
        latent_data_df['FuzzyCMeans_Labels'] = cluster_labels
        results['FuzzyCMeans_Labels'] += calculate_success_range(data_success, latent_data_df['FuzzyCMeans_Labels'], 'FuzzyCMeans_Labels')

        # Random Clustering
        cluster_labels = np.random.randint(0, num_clusters, size=len(data))
        latent_data_df['Random_Labels'] = cluster_labels
        results['Random_Labels'] += calculate_success_range(data_success, latent_data_df['Random_Labels'], 'Random_Labels')

    # Create results DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Success Range'])
    results_df['Success Range'] = results_df['Success Range'] / n  # Average over n seeds

    return results_df

def test_dimensions():
    lst = []
    for i in range(1,43):
        seed = i
        np.random.seed(seed)
        torch.manual_seed(seed)
        latent_data_df = encode(data, num_epochs = 50, dim = 10)
        scaler = StandardScaler()
        latent_data_scaled = scaler.fit_transform(latent_data_df)
        # GMM Clustering
        sum = 0
        for j in range(15):
            gmm = GaussianMixture(n_components=5, random_state=j)
            latent_data_df['GMM_Labels'] = gmm.fit_predict(latent_data_scaled)
            sum += calculate_success_range(data_success, latent_data_df['GMM_Labels'], 'GMM_Labels')
        lst.append(sum/15)
    x_range = range(1, 43)
    plt.plot(x_range, lst)
    plt.xlabel('Number of Latent Dimensions')
    plt.ylabel('Average Success Range')
    plt.title('Average Success Range vs. Number of Latent Dimensions')
    plt.show()


def best_cluster_count_silhoutte(data):
    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = []
    clusters_range = range(AnalysisConfig.min_clusters, AnalysisConfig.max_clusters + 1)

    for n_clusters in clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=40)
        cluster_labels = gmm.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    silhouette_df = pd.DataFrame({
        "Clusters": list(clusters_range),
        "Silhouette Score": silhouette_scores
    })
    return int(silhouette_df.iloc[silhouette_df['Silhouette Score'].idxmax()]['Clusters'])


def best_subcluster_count_davies_bouldin(latent_data_df, best_cluster_count, cluster_labels):
    latent_data_df_clusters = latent_data_df.copy()
    latent_data_df_clusters['Cluster'] = cluster_labels

    best_cluster_count_lst = []

    for i in range(1 ,best_cluster_count + 1):
        # Step 1: Prepare Latent Data
        latent_data_i = latent_data_df_clusters[latent_data_df_clusters['Cluster'] == i].values

        # Scale the data to ensure consistent distance calculation
        scaler = StandardScaler()
        scaled_latent_data_i = scaler.fit_transform(latent_data_i)

        # Step 3: Generate Clusters for Different Numbers (5-10)
        cluster_range = range(AnalysisConfig.min_subclusters, AnalysisConfig.max_subclusters)
        scores = {}

        for num_clusters in cluster_range:
            clusters_i = GaussianMixture(n_components=num_clusters, random_state=42).fit_predict(scaled_latent_data_i)
            score = davies_bouldin_score(latent_data_i, clusters_i)
            scores[num_clusters] = score

        # Step 4: Plot davies_bouldin_scores
        #plt.figure(figsize=(8, 5))
        #plt.plot(list(davies_bouldin_scores.keys()), list(davies_bouldin_scores.values()), marker='o')
        #plt.title("davies_bouldin_scores for Different Cluster Numbers")
        #plt.xlabel("Number of Clusters")
        #plt.ylabel("davies_bouldin_score")
        #plt.grid()
        #plt.show()

        # Return davies_bouldin_scores
        scores_df = pd.DataFrame({
            "Number of Clusters": list(scores.keys()),
            "DB Scores": list(scores.values())
        })
        best_cluster_count_lst.append(int(scores_df.iloc[scores_df['DB Scores'].idxmin()]['Number of Clusters']))
    return best_cluster_count_lst


def gmm_cluster_with_min_size(data, n_clusters, min_size, random_state=42, max_iterations=100):
    """
    Fits a Gaussian Mixture Model with n_clusters components on 'data'.
    Reassigns points in small clusters (< min_size) to other clusters until
    all clusters meet the size constraint or max_iterations is reached.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        The data to cluster.
    n_clusters : int
        Number of clusters to fit in the GMM.
    min_size : int
        Minimum size for each cluster.
    random_state : int
        Seed for the random number generator.
    max_iterations : int
        Maximum number of iterations to enforce the size constraint.
        
    Returns:
    --------
    labels : array, shape (n_samples,)
        Final cluster assignments for each data point.
    gmm : GaussianMixture object
        The trained GMM model.
    """

    # 1) Fit the data with a GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm.fit(data)

    # 2) Initial cluster assignment
    labels = gmm.predict(data)

    # 3) Iteratively check and fix cluster sizes
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Count cluster sizes
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        
        # Identify small clusters
        small_clusters = np.where(cluster_counts < min_size)[0]

        # If no small clusters, we are done
        if len(small_clusters) == 0:
            #print(f"Converged after {iteration} iterations.")
            break

        for cluster_id in small_clusters:
            # Indices of points in the small cluster
            idx_small_cluster = np.where(labels == cluster_id)[0]

            # Skip empty clusters
            if len(idx_small_cluster) == 0:
                continue

            # Posterior probabilities for these points across all clusters
            responsibilities = gmm.predict_proba(data[idx_small_cluster])

            # Reassign points to other clusters based on highest probability
            for i, idx in enumerate(idx_small_cluster):
                # Find the best cluster for this point (exclude current cluster)
                probs = responsibilities[i]
                probs[cluster_id] = -1  # Exclude current cluster by setting probability to -1
                new_cluster = np.argmax(probs)
                labels[idx] = new_cluster

        # Debug: Print cluster sizes after reassignment
        #print(f"Iteration {iteration}: Cluster sizes = {np.bincount(labels, minlength=n_clusters)}")

    # Check if max iterations reached
    if iteration == max_iterations:
        print("Warning: Maximum iterations reached. Some clusters may still be smaller than min_size.")

    # 4) Return final cluster labels and GMM
    return labels + 1, gmm

def gmm_subclusters_with_min_size(latent_data_df):
    # Step 1: Make a copy and initialize SubCluster
    latent_data_df_clusters = latent_data_df.copy()
    latent_data_df_clusters['Cluster'] = cluster_labels
    latent_data_df_subclusters = latent_data_df_clusters.copy()
    latent_data_df_subclusters['SubCluster'] = 0

    gmm_subclusters = []

    for i in range(1, len(best_subcluster_count_lst) + 1):
        # Filter data for the current cluster
        cluster_data = latent_data_df_clusters.loc[latent_data_df_clusters['Cluster'] == i]

        # Scale the data
        scaler = StandardScaler()
        scaled_cluster_data = scaler.fit_transform(cluster_data.drop(columns=['Cluster']))

        # Perform GMM clustering
        sub_clusters, gmm_subcluster_i = gmm_cluster_with_min_size(scaled_cluster_data, best_subcluster_count_lst[i - 1], AnalysisConfig.min_cluster_size)

        gmm_subclusters.append(gmm_subcluster_i)
        
        # Assign SubCluster labels back to the main DataFrame
        latent_data_df_subclusters.loc[latent_data_df_subclusters['Cluster'] == i, 'SubCluster'] = sub_clusters
    return latent_data_df_subclusters, gmm_subclusters

def create_success_rate_table(latent_data_df_subclusters):
    data['Cluster'] = latent_data_df_subclusters['Cluster']
    data['SubCluster'] = latent_data_df_subclusters['SubCluster']
    data_success['Cluster'] = latent_data_df_subclusters['Cluster']
    data_success['SubCluster'] = latent_data_df_subclusters['SubCluster']

    # Step 1: Calculate success rates for clusters and subclusters
    cluster_success = data_success.groupby('Cluster')['success'].mean().reset_index()
    cluster_success.columns = ['Cluster', 'success_rate']
    cluster_size = data_success['Cluster'].value_counts().reset_index()
    cluster_size.columns = ['Cluster', 'cluster_size']

    subcluster_success = data_success.groupby(['Cluster', 'SubCluster'])['success'].mean().reset_index()
    subcluster_success.columns = ['Cluster', 'SubCluster', 'success_rate']
    subcluster_size = data_success.groupby(['Cluster', 'SubCluster']).size().reset_index(name='subcluster_size')

    # Merge cluster and subcluster data
    clusters_combined = pd.merge(cluster_success, cluster_size, on='Cluster')
    subclusters_combined = pd.merge(subcluster_success, subcluster_size, on=['Cluster', 'SubCluster'])

    # Step 2: Rank clusters by overall success rate and relabel them
    clusters_combined = clusters_combined.sort_values(by='success_rate', ascending=False).reset_index(drop=True)
    clusters_combined['New Cluster ID'] = ['Cluster {}'.format(i+1) for i in range(len(clusters_combined))]

    # Create a mapping from old cluster to new cluster names
    cluster_map = dict(zip(clusters_combined['Cluster'], clusters_combined['New Cluster ID']))

    # Step 3: Relabel subclusters based on the new cluster ordering and sort subclusters
    updated_table = []

    for i, cluster_row in clusters_combined.iterrows():
        new_cluster_id = cluster_row['New Cluster ID']
        updated_table.append({
            'Type': 'Main',
            'Cluster ID': new_cluster_id,
            'Success Rate': cluster_row['success_rate'],
            'Cluster Size': cluster_row['cluster_size']
        })

        # Filter and relabel subclusters
        subclusters = subclusters_combined[subclusters_combined['Cluster'] == cluster_row['Cluster']]
        subclusters = subclusters.sort_values(by='success_rate', ascending=False).reset_index(drop=True)

        for j, sub_row in subclusters.iterrows():
            updated_table.append({
                'Type': 'Sub',
                'Cluster ID': f"{new_cluster_id}_{j+1}",
                'Success Rate': sub_row['success_rate'],
                'Cluster Size': sub_row['subcluster_size']
            })

    # Convert the results to a DataFrame
    success_table = pd.DataFrame(updated_table)
    success_table['Normalized Success Rate'] = success_table['Success Rate']*k
    return success_table


def create_final_df(data_success, latent_data_df_subclusters):
    data_success['Cluster'] = latent_data_df_subclusters['Cluster']
    data_success['SubCluster'] = latent_data_df_subclusters['SubCluster']
    # Step 1: Compute the average success rate for each cluster and sort them
    data_success['Cluster'] = data_success['Cluster'].astype(str)
    data_success['SubCluster'] = data_success['SubCluster'].astype(str)

    cluster_success = data_success.groupby('Cluster')['success'].mean().sort_values(ascending=False)
    cluster_mapping = {cluster: idx + 1 for idx, cluster in enumerate(cluster_success.index)}

    # Step 2: Assign new cluster IDs based on the mapping
    data_success['cluster_ID'] = data_success['Cluster'].map(cluster_mapping)

    # Step 3: Compute average success rate for each subcluster within each cluster
    subcluster_success = (
        data_success.groupby(['Cluster', 'SubCluster'])['success']
        .mean()
        .reset_index()
        .sort_values(by=['Cluster', 'success'], ascending=[True, False])
    )

    # Step 4: Assign new subcluster IDs within each cluster
    subcluster_success['subcluster_ID'] = subcluster_success.groupby('Cluster').cumcount() + 1

    # Step 5: Merge the subcluster IDs back into the main dataset
    data_success = data_success.merge(
        subcluster_success[['Cluster', 'SubCluster', 'subcluster_ID']],
        on=['Cluster', 'SubCluster'],
        how='left'
    )

    data_success.drop(columns=['Cluster', 'SubCluster', 'success'], inplace=True)
    return data_success

def predict_cluster(data, encoding_scaler, vae, gmm_cluster):
    formatted_data = pd.DataFrame(data).T
    # Step 1: Encode the new data point
    encoded_point = encode_new_data(formatted_data, encoding_scaler, vae)

    # Step 2: Predict the cluster using the GMM model
    cluster = gmm_cluster.predict(encoded_point)[0]

    return cluster

def predict_cluster_full(data, encoding_scaler, vae, gmm_cluster, gmm_subclusters):
    """
    Predict the cluster and subcluster probabilities for given data.

    Parameters:
        data (pd.DataFrame or np.ndarray): Input data points to predict.
        encoding_scaler (StandardScaler): Scaler used for feature scaling.
        vae: Variational Autoencoder used for encoding.
        gmm_cluster (GaussianMixture): GMM model for main clusters.
        gmm_subclusters (list of GaussianMixture): List of GMM models for subclusters.

    Returns:
        pd.DataFrame: DataFrame with probabilities for each cluster-subcluster pair
                      and total probabilities for each main cluster.
    """
    # Format and preprocess data
    formatted_data = pd.DataFrame(data).T

    # Step 1: Encode the new data point
    encoded_point = encode_new_data(formatted_data, encoding_scaler, vae)

    # Step 2: Predict the cluster using the main GMM model
    cluster_probs = gmm_cluster.predict_proba(encoded_point)[0]
    predicted_cluster = gmm_cluster.predict(encoded_point)[0]

    # Initialize a dictionary to store probabilities
    result = {
        "Cluster": [],
        "Subcluster": [],
        "Probability": []
    }

    # Step 3: Calculate probabilities for subclusters within each cluster
    for cluster_idx, sub_gmm in enumerate(gmm_subclusters):
        if cluster_probs[cluster_idx] > 0:  # Consider only non-zero probability clusters
            # Get subcluster probabilities
            subcluster_probs = sub_gmm.predict_proba(encoded_point)[0]
            for subcluster_idx, prob in enumerate(subcluster_probs):
                result["Cluster"].append(cluster_idx)
                result["Subcluster"].append(subcluster_idx)
                result["Probability"].append(np.round(cluster_probs[cluster_idx] * prob,4))

    # Convert the result dictionary to a DataFrame
    result_df = pd.DataFrame(result)

    # Step 4: Add total probabilities for each main cluster
    total_cluster_probs = pd.DataFrame({
        "Cluster": np.arange(len(cluster_probs)),
        "Total_Cluster_Probability": np.round(cluster_probs,4)
    })

    # Merge the subcluster probabilities with total cluster probabilities
    final_df = result_df.merge(total_cluster_probs, on="Cluster", how="left")

    return final_df

if __name__ == '__main__':

    # Step 1: Data Preparation
    data_unclean = pd.read_csv('(December 2024)_ Founders data - feature_engineered.csv')

    data = clean(data_unclean)
    data_success = clean(data_unclean, remove_success=False)
    #Scaling Rate Compared to Real Life
    k = 0.019/(sum(data_unclean['success'])/len(data_unclean))

    if AnalysisConfig.encoded:
        latent_data_df, encoding_scaler, vae = encode(data, AnalysisConfig.num_epochs, dim = 10)
        latent_data = latent_data_df.values
    else:
        latent_data_df = data
        latent_data = latent_data_df.values

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(latent_data_df)

    # Perform GMM clustering with constraints
    best_cluster_count = best_cluster_count_silhoutte(data_scaled)

    cluster_labels, gmm_cluster = gmm_cluster_with_min_size(data_scaled, best_cluster_count, AnalysisConfig.min_cluster_size)

    best_subcluster_count_lst = best_subcluster_count_davies_bouldin(latent_data_df, best_cluster_count, cluster_labels)

    latent_data_df_subclusters, gmm_subclusters = gmm_subclusters_with_min_size(latent_data_df)

    #Create final tables
    success_table = create_success_rate_table(latent_data_df_subclusters)

    final_df = create_final_df(data_success, latent_data_df_subclusters)

    #Predict cluster example
    data = clean(data_unclean)
    new_founder = data.iloc[0]

    cluster_probabilities = predict_cluster(new_founder, encoding_scaler, vae, gmm_cluster)

    full_cluster_subcluster_probabilities = predict_cluster_full(new_founder, encoding_scaler, vae, gmm_cluster, gmm_subclusters)

    print(full_cluster_subcluster_probabilities)
