import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import seaborn as sns


# Load the pkl file
def make_analysis(group,n_clusters):

    with open('Treated_'+str(group)+'.pkl', mode='rb') as f:
        error = pickle.load(f)
        X = error['X_test']
        y = error['y_test']
        g = error['g_test']
        bios = error['bio_test']
        y_pred = error['predicted_job']
        jobid_2_job = error['jobid_2_job']

    job_ids = np.argmax(y, axis=1)
    jobs = [jobid_2_job[i] for i in job_ids]
    pd.DataFrame({'predicted job' : y_pred, 'true job' : job_ids}).to_csv('predicted_vs_true_job_'+str(group)+'.csv')

    # plot number of errors per job
    nbr_errors_true = np.sum(y, axis = 0) 
    plt.figure()
    plt.bar([jobid_2_job[i] for i in np.arange(len(nbr_errors_true))], nbr_errors_true)
    plt.xlabel('True Job')
    plt.xticks(rotation=90)
    plt.ylabel('Number of errors')
    plt.title('Number of errors per true job for group ' + str(group))
    plt.savefig('errors_per_job_'+str(group)+'.pdf')

    # plot number of errors per predicted job
    nbr_errors_pred = np.bincount(y_pred)
    plt.figure()
    plt.bar([jobid_2_job[i] for i in np.arange(len(nbr_errors_pred))], nbr_errors_pred)
    plt.xlabel('Predicted Job')
    plt.xticks(rotation=90)
    plt.ylabel('Number of errors')
    plt.title('Number of errors per predicted job for group ' + str(group))
    plt.savefig('errors_per_predicted_job_'+str(group)+'.pdf')


    # Load the DistilBert model and tokenizer
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)

    def get_embeddings(list_input_texts, tokenizer, Max_len):
        embeddings = []
        print('Computing embeddings')
        for input_text in tqdm(list_input_texts):
            inputs = tokenizer.encode_plus(
                input_text,
                None,
                add_special_tokens=True,
                max_length=Max_len,
                return_token_type_ids=True,
                padding='max_length',
                truncation=True
            )
            ids = torch.tensor([inputs['input_ids']], dtype=torch.long)
            mask = torch.tensor([inputs['attention_mask']], dtype=torch.long)

            # Get the BERT model's output
            with torch.no_grad():
                output = model(ids, mask)

            # The final hidden state of the [CLS] token can be used as a representation of the text
            cls_output = output[0][:,0,:].numpy()
            print(cls_output.shape)

            # compute the average of the embeddings, weighted by masks
            #mask = mask.numpy()
            #mask = mask.flatten()
            #cls_output = cls_output * mask[:, np.newaxis]
            #embeds = np.sum(cls_output, axis=0) / np.sum(mask)
            embeddings.append(cls_output)

        return embeddings

    # Get embeddings for all input sentences
    embeddings = get_embeddings(bios, tokenizer, Max_len=512)
    np.savetxt("embeddings_clstoken_"+str(group)+".csv", embeddings, delimiter=",")
    # read from .csv if saved
    #embeddings = pd.read_csv('embeddings_clstoken_'+str(group)+'.csv', header=None)
    print('Embeddings shape : ',embeddings.shape)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Calculate the point density
    xy = np.vstack([tsne_embeddings[:,0],tsne_embeddings[:,1]])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = tsne_embeddings[:,0][idx], tsne_embeddings[:,1][idx], z[idx]

    plt.figure(figsize=(12,8))
    sns.scatterplot(x=x, y=y, hue=z, palette="plasma", alpha=0.9)
    sns.kdeplot(x=x,y=y,levels=10,fill=True,linewidths=1, cmap = "plasma", alpha=0.7)
    plt.title('TSNE scatter-density plot of the embeddings for group ' + str(group))
    plt.savefig('tsne_'+str(group)+'.pdf')

    # Reduce the dimensionality of the embeddings
    pca = PCA(n_components=5)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_embeddings)

    # Get the cluster labels for each bio
    labels = kmeans.labels_
    centroids_reduced = kmeans.cluster_centers_
    centroids = pca.inverse_transform(centroids_reduced)

    # Find the closest data points to the centroids
    n_split = 1
    #workrelated_words_1 = ["Outsiders", "Women", "Creatives", "Caregivers", "Minorities",
                        #"Immigrants", "Elderly", "LGBTQ+", "Disabled",  "Religious", "Racial", 
                        #"Ethnic", "Indigenous", "Youth", "Refugees", "Homeless", "Low-Income", 
                        #"Nonconformists", "Introverts", "Overweight", "Underprivileged", "Rich",
                        #"Unemployed", "Divorced", "Non-religious", "Mother",
                        #"Non-native", "Non-binary", "Transgender", "Intersectional", "Working-class",
                        #"Disadvantaged", "Minorities", "Isolated", "Marginalized",  "Sex",
                        #"Non-citizens", "Children", "Student", "PhD", "Graduate", "Undergraduate"]
    workrelated_words_1 = ["Sex","Ethnic","Age","Parent","Divorced","PhD","Rich","Health","Religion",
                           "Veteran","Gender","Race","Old","Married","Poor","Handicap"]
    #workrelated_words_2 = ["Gender","Race","Old","Care","Married","Poor","Gay","Health","Handicap"]
    #workrelated_words_3 = ["Teamwork", "Inclusive", "Innovative", "Flexible", "Young", "Remote", "Digital", "Adaptable", "Resilient", "Experience", "Empathetic", "Tech", "Diverse"]
    workrelated_words_list = np.array([workrelated_words_1])
    #np.random.shuffle(workrelated_words)
    #workrelated_words_list = np.array_split(workrelated_words, n_split)
    print("Number of words in split : ",len(workrelated_words_list[0]))
    print("Number of splits : ",len(workrelated_words_list))
    print("Computing splits")
    centroid_keywords = np.full((n_clusters,n_split),"", dtype="<U20")

    for k in range(n_split):
        workrelated_words_split=list(workrelated_words_list[k])
        keywords_embeddings = get_embeddings(workrelated_words_split, tokenizer, Max_len=512)
        reduced_keywords_embeddings = pca.transform(keywords_embeddings)

        #pca_2d = PCA(n_components=2)  # Ensure 2D for visualization
        #reduced_keywordsembeddings_2d = pca_2d.fit_transform(keywords_embeddings)
        #reduced_embeddings_2d = pca_2d.transform(embeddings)

        # Compute the Voronoi partition
        #vor = Voronoi(reduced_keywordsembeddings_2d)

        # Plot the Voronoi diagram
        #fig = voronoi_plot_2d(vor)
        #plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue')
        # setting limits for the axes
        #plt.xlim(-3, 4)
        #plt.ylim(-3.5, 3.5)
        #plt.savefig('voronoi_'+str(group)+'.pdf')

        closest_keywords_id = pairwise_distances_argmin(centroids_reduced, reduced_keywords_embeddings, metric = "manhattan")
        closest_keywords_id_indiv = pairwise_distances_argmin(reduced_embeddings, reduced_keywords_embeddings, metric = "manhattan")
        closest_keywords_indiv = [workrelated_words_split[closest_keywords_id_indiv[j]] for j in range(len(closest_keywords_id_indiv))]
        for n in tqdm(range(n_clusters)):
            # get the closest keyword to the centroid
            closest_keyword = workrelated_words_split[closest_keywords_id[n]]
            centroid_keywords[n][k] = closest_keyword
            

    # save centroids, keywords, various infos to csv
    closest_points = pd.DataFrame(columns = ['cluster','cluster keyword', 'individual keyword', 'sex', 'job','predicted job', 'bio'])
    whole_clusters = pd.DataFrame(columns = ['cluster','cluster keyword', 'individual keyword', 'sex', 'job','predicted job', 'bio'])
    NN_keywords = NearestNeighbors(n_neighbors=12, algorithm='ball_tree').fit(reduced_embeddings)

    closest_bios_to_keywords = pd.DataFrame(columns = ['keyword', 'bio'])
    for i in range(len(keywords_embeddings)) :
        distances, indices = NN_keywords.kneighbors([reduced_keywords_embeddings[i]])
        #print(workrelated_words_1[i], reduced_keywords_embeddings[i], distances)
        
        for j in range(12):
            closest_bios_to_keywords = closest_bios_to_keywords.append({'keyword': workrelated_words_1[i], 'bio': bios[indices[0][j]]}, ignore_index=True)
    
    closest_bios_to_keywords.to_csv('closest_bios_to_keywords_'+str(group)+'.csv')

    print('Computing clusters')
    for i in tqdm(range(n_clusters)):
        keywords = list(centroid_keywords[i])[0]
        cluster_bios = [bios[j] for j in range(len(bios)) if labels[j] == i]
        cluster_embeddings = [reduced_embeddings[j] for j in range(len(bios)) if labels[j] == i]
        cluster_jobs = [jobs[j] for j in range(len(bios)) if labels[j] == i]
        cluster_predicted_jobs = [jobid_2_job[y_pred[j].item()] for j in range(len(bios)) if labels[j] == i]
        cluster_gender = [g[j] for j in range(len(bios)) if labels[j] == i]
        cluster_indivkeywords = [closest_keywords_indiv[j] for j in range(len(bios)) if labels[j] == i]

        # Create a NearestNeighbors object
        NN_centroids = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(cluster_embeddings)
        
        # Find the 10 closest points to the centroid
        distances, indices = NN_centroids.kneighbors([centroids_reduced[i]])
        
        #add the 10 points (cluster, keyword, job, predicted job, bio) to the dataframe
        for j in range(10):
            closest_points = closest_points.append({'cluster': i, 'cluster keyword': keywords, 'individual keyword' : cluster_indivkeywords[indices[0][j]], 'job': cluster_jobs[indices[0][j]], 'predicted job' : cluster_predicted_jobs[indices[0][j]] , 'bio': cluster_bios[indices[0][j]]}, ignore_index=True)

        #add the all points to the whole_clusters dataframe
        for id in range(len(cluster_bios)):
            whole_clusters = whole_clusters.append({'cluster': i, 'cluster keyword': keywords, 'individual keyword' : cluster_indivkeywords[id], 'sex' : cluster_gender[id] , 'job': cluster_jobs[id], 'predicted job' : cluster_predicted_jobs[id] , 'bio': cluster_bios[id]}, ignore_index=True)

    # save closest points to csv
    closest_points.to_csv('closest_points_'+str(group)+'.csv')
    whole_clusters.to_csv('whole_clusters_'+str(group)+'.csv')
    

def save_clustered_bios(clusters_to_bios, group):
    with open('clustered_bios_'+str(group)+'.pkl', 'wb') as f:
        pickle.dump(clusters_to_bios, f)


#save_clustered_bios(make_analysis(0,10), 0)
#save_clustered_bios(make_analysis(1,10), 1)
#save_clustered_bios(make_analysis('_ErrorGlobal',5), '_ErrorGlobal')
make_analysis('alltest',100)
