import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors


# Load the pkl file
def make_analysis(group,n_clusters):

    with open('Treated_'+str(group)+'.pkl', mode='rb') as f:
        error = pickle.load(f)
        X = error['X_test']
        y = error['y_test']
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
            cls_output = output[0][0].numpy()
            # compute the average of the embeddings, weighted by masks
            mask = mask.numpy()
            mask = mask.flatten()
            cls_output = cls_output * mask[:, np.newaxis]
            embeds = np.sum(cls_output, axis=0) / np.sum(mask)
            embeddings.append(embeds)

        return embeddings

    # Get embeddings for all input sentences
    #embeddings = get_embeddings(bios, tokenizer, Max_len=512)
    #np.savetxt("embeddings_"+str(group)+".csv", embeddings, delimiter=",")
    # read from .csv if saved
    embeddings = pd.read_csv('embeddings_'+str(group)+'.csv', header=None).values

    # Reduce the dimensionality of the embeddings
    pca = PCA(n_components=10)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_embeddings)

    # Get the cluster labels for each bio
    labels = kmeans.labels_
    centroids_reduced = kmeans.cluster_centers_
    centroids = pca.inverse_transform(centroids_reduced)

    # Find the closest data points to the centroids
    workrelated_words = ["Outsiders", "Women", "Creatives", "Caregivers", "Minorities", "Immigrants", "Elderly", "LGBTQ+", "Disabled",  "Religious", "Racial", "Ethnic", "Indigenous", "Single Parents", "Youth", "Refugees", "Homeless", "Educationally Disadvantaged", "Low-Income", "Nonconformists", "Mental Health", "Introverts", "Ex-Convicts", "Overweight", "Underprivileged", "Unemployed", "Displaced", "Unwed Mothers", "Divorced", "Non-religious", "Non-native Speakers", "Non-heteronormative", "Non-binary", "Transgender", "Intersectional", "Working-class", "Non-cisnormative", "Economically Disadvantaged", "Differently-gendered", "Sexual Minorities", "Cultural Minorities", "Socially Isolated", "Economically Marginalized", "Stigmatized", "Non-traditional Families", "Non-English Speakers", "Non-citizens", "Undocumented", "Alternative Lifestyles", "Non-affiliated"]    
    keywords_embdeddings = get_embeddings(workrelated_words, tokenizer, Max_len=512)

    closest_keywords, _ = pairwise_distances_argmin_min(centroids, keywords_embdeddings)
    centroids_keywords = [workrelated_words[i] for i in closest_keywords]  

    closest_points = pd.DataFrame(columns = ['cluster','keyword', 'job','predicted job', 'bio'])
    whole_clusters = pd.DataFrame(columns = ['cluster','keyword', 'job','predicted job', 'bio']) 

    print('Computing clusters')
    for i in tqdm(range(n_clusters)):
        cluster_bios = [bios[j] for j in range(len(bios)) if labels[j] == i]
        cluster_embeddings = [embeddings[j] for j in range(len(bios)) if labels[j] == i]
        cluster_jobs = [jobs[j] for j in range(len(bios)) if labels[j] == i]
        cluster_predicted_jobs = [jobid_2_job[y_pred[j].item()] for j in range(len(bios)) if labels[j] == i]

        # Create a NearestNeighbors object
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(cluster_embeddings)
        
        # Find the 20 closest points to the centroid
        distances, indices = nbrs.kneighbors([centroids[i]])
        
        #add the 20 points (cluster, keyword, job, predicted job, bio) to the dataframe
        for j in range(20):
            closest_points = closest_points.append({'cluster': i, 'keyword': centroids_keywords[i], 'job': cluster_jobs[indices[0][j]], 'predicted job' : cluster_predicted_jobs[indices[0][j]] , 'bio': cluster_bios[indices[0][j]]}, ignore_index=True)

        #add the all points to the whole_clusters dataframe
        for id in range(len(cluster_bios)):
            whole_clusters = whole_clusters.append({'cluster': i, 'keyword': centroids_keywords[i], 'job': cluster_jobs[id], 'predicted job' : cluster_predicted_jobs[id] , 'bio': cluster_bios[id]}, ignore_index=True)


    # save closest points to csv
    closest_points.to_csv('closest_points_'+str(group)+'.csv')
    whole_clusters.to_csv('whole_clusters_'+str(group)+'.csv')
    

def save_clustered_bios(clusters_to_bios, group):
    with open('clustered_bios_'+str(group)+'.pkl', 'wb') as f:
        pickle.dump(clusters_to_bios, f)


#save_clustered_bios(make_analysis(0,10), 0)
#save_clustered_bios(make_analysis(1,10), 1)
#save_clustered_bios(make_analysis('_ErrorGlobal',5), '_ErrorGlobal')
make_analysis('alltest',5)