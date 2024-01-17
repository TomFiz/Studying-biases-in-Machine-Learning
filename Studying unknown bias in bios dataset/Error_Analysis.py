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

    with open('Treated_Error'+str(group)+'.pkl', mode='rb') as f:
        error = pickle.load(f)
        X = error['X_test']
        y = error['y_test']
        bios = error['bio_test']
        y_pred = error['predicted_job']
        jobid_2_job = error['jobid_2_job']

    job_ids = np.argmax(y, axis=1)
    jobs = [jobid_2_job[i] for i in job_ids]
    pd.DataFrame({'predicted job' : y_pred, 'true job' : job_ids}).to_csv('predicted_vs_true_job'+str(group)+'.csv')

    # plot number of errors per job
    nbr_errors_true = np.sum(y, axis = 0) 
    plt.figure()
    plt.bar([jobid_2_job[i] for i in np.arange(len(nbr_errors_true))], nbr_errors_true)
    plt.xlabel('True Job')
    plt.xticks(rotation=90)
    plt.ylabel('Number of errors')
    plt.title('Number of errors per true job for group ' + str(group))
    plt.savefig('errors_per_job'+str(group)+'.pdf')

    # plot number of errors per predicted job
    nbr_errors_pred = np.bincount(y_pred)
    plt.figure()
    plt.bar([jobid_2_job[i] for i in np.arange(len(nbr_errors_pred))], nbr_errors_pred)
    plt.xlabel('Predicted Job')
    plt.xticks(rotation=90)
    plt.ylabel('Number of errors')
    plt.title('Number of errors per predicted job for group ' + str(group))
    plt.savefig('errors_per_predicted_job'+str(group)+'.pdf')


    # Load the DistilBert model and tokenizer
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)

    def get_embeddings(list_input_texts, tokenizer, Max_len):
        embeddings = []
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
    embeddings = get_embeddings(bios, tokenizer, Max_len=512)
    #save embeddings as csv
    np.savetxt("embeddings"+str(group)+".csv", embeddings, delimiter=",")
    #embeddings = pd.read_csv('embeddings'+str(group)+'.csv', header=None).values


    pca = PCA(n_components=10)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters)  # Use 5 clusters as an example
    kmeans.fit(reduced_embeddings)


    # Get the cluster labels for each bio
    labels = kmeans.labels_
    centroids_reduced = kmeans.cluster_centers_
    centroids = pca.inverse_transform(centroids_reduced)

    # Find the closest data points to the centroids
    workrelated_words = ["Age", "Gender", "Education level", "Income", "Geographic location", "Ethnicity", "Occupation", "Marital status", "Health status", "Religion",  "Political affiliation", "Social network size", "Internet usage", "Technology adoption", "Language proficiency", "Personality traits", "Cognitive abilities", "Emotional intelligence", "Physical fitness", "Nutrition habits",  "Sleep patterns", "Employment status", "Job satisfaction", "Job performance", "Work-life balance", "Stress levels", "Family structure", "Parental status", "Housing situation", "Transportation mode", "Hobbies and interests",  "Media consumption", "Social media activity", "Relationship status", "Communication style", "Learning preferences", "Cultural background", "Environmental concerns", "Financial literacy", "Risk tolerance", "Attitudes toward authority",  "Decision-making style", "Creativity", "Problem-solving skills", "Resilience", "Coping mechanisms", "Self-esteem", "Trust in institutions", "Civic engagement", "Volunteerism", "Economic beliefs",  "Entrepreneurial mindset", "Attitudes toward change", "Open-mindedness", "Emotional resilience", "Empathy", "Social support network", "Substance use", "Legal history", "Criminal record", "Access to healthcare",  "Health behaviors", "Genetic factors", "Environmental exposures", "Academic achievement", "Learning disabilities", "Reading habits", "Math skills", "Science literacy", "Technological literacy", "Digital skills",  "Information-seeking behavior", "Communication skills", "Interpersonal skills", "Teamwork skills", "Leadership skills", "Cultural competency", "Time management", "Financial management skills", "Decision-making processes", "Risk management",  "Conflict resolution skills", "Negotiation skills", "Communication technology usage", "Privacy concerns", "Data security awareness", "Trust in technology", "Attitudes toward automation", "Environmental awareness", "Recycling habits", "Sustainable practices",  "Political awareness", "Civic responsibility", "Voting behavior", "Political participation", "Legal awareness", "Consumer behavior", "Brand loyalty", "Advertising responsiveness", "Social influence susceptibility", "Peer pressure resistance",  "Educational aspirations", "Career ambitions", "Job preferences", "Work ethic", "Time management", "Procrastination tendencies", "Learning motivation", "Achievement motivation", "Goal-setting behaviors", "Resilience to failure",  "Perception of success", "Risk-taking propensity", "Financial risk-taking", "Sports involvement", "Physical activity preferences", "Team sports vs. individual sports", "Competitive spirit", "Sportsmanship", "Athletic ability", "Recreational activities",  "Travel preferences", "Adventure-seeking behavior", "Cultural experiences", "Food preferences", "Dietary restrictions", "Culinary skills", "Cooking habits", "Food shopping behaviors", "Eating out frequency", "Fast food consumption",  "Alcohol consumption", "Substance use attitudes", "Substance abuse history", "Gambling behaviors", "Gaming habits", "Technology addiction", "Social networking addiction", "Television watching habits", "Movie preferences", "Music preferences",  "Artistic tastes", "Literature preferences", "Reading habits", "Educational content consumption", "News consumption", "Political news engagement", "Conspiracy beliefs", "Religious beliefs", "Spiritual practices", "Atheism vs. theism",  "Belief in the supernatural", "Paranormal beliefs", "Philosophical views", "Existential beliefs", "Moral values", "Ethical decision-making", "Altruism", "Selfishness", "Empathy", "Cooperation vs. competition", "Trust in others",  "Social capital", "Social exclusion sensitivity", "Loneliness", "Relationship satisfaction", "Communication patterns in relationships", "Conflict resolution in relationships", "Intimacy preferences", "Romantic vs. platonic relationships", "Relationship commitment",  "Online dating preferences", "Sexual orientation", "Gender identity", "Body image satisfaction", "Self-perception of attractiveness", "Fashion preferences", "Grooming habits", "Tattoos and piercings", "Physical disabilities", "Mental health history",  "Mental health treatment seeking behavior", "Therapy preferences", "Stigma around mental health", "Coping strategies", "Self-harm tendencies", "Suicidal ideation", "Trauma history", "PTSD symptoms", "Personality disorders", "Mood disorders",  "Anxiety disorders", "Substance use disorders", "Eating disorders", "Sleep disorders", "Neurological disorders", "Chronic health conditions", "Allergies", "Medication usage", "Alternative medicine usage", "Health insurance status", "Preventive healthcare practices",  "Vaccination beliefs", "Access to healthcare services", "Health literacy", "Physical activity level", "Exercise preferences", "Fitness goals", "Body composition goals", "Sleep hygiene", "Leisure activities", "Outdoor vs. indoor activities",  "Nature exposure preferences", "Environmental conservation attitudes", "Green living practices", "Homeownership vs. renting", "Living space preferences", "Commuting preferences", "Mode of transportation", "Car ownership", "Public transportation usage",  "Biking habits", "Walking habits", "Technology adoption rate", "Social media platform preferences", "Online shopping habits", "Payment method preferences", "Financial decision-making", "Budgeting habits", "Investment preferences", "Retirement planning","Financial risk tolerance", "Entrepreneurial"]
    keywords_embdeddings = get_embeddings(workrelated_words, tokenizer, Max_len=512)

    closest_keywords, _ = pairwise_distances_argmin_min(centroids, keywords_embdeddings)
    centroids_keywords = [workrelated_words[i] for i in closest_keywords]  

    closest_points = pd.DataFrame(columns = ['cluster','keyword', 'job','predicted job', 'bio'])
    for i in range(n_clusters):
        cluster_bios = [bios[j] for j in range(len(bios)) if labels[j] == i]
        cluster_embeddings = [embeddings[j] for j in range(len(bios)) if labels[j] == i]
        cluster_jobs = [jobs[j] for j in range(len(bios)) if labels[j] == i]
        cluster_predicted_jobs = [jobid_2_job[y_pred[j].item()] for j in range(len(bios)) if labels[j] == i]

        # Create a NearestNeighbors object
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(cluster_embeddings)
        
        # Find the 5 closest points to the centroid
        distances, indices = nbrs.kneighbors([centroids[i]])
        
        #add the 5 points (cluster, keyword, job, predicted job, bio) to the dataframe
        for j in range(5):
            closest_points = closest_points.append({'cluster': i, 'keyword': centroids_keywords[i], 'job': cluster_jobs[indices[0][j]], 'predicted job' : cluster_predicted_jobs[indices[0][j]] , 'bio': cluster_bios[indices[0][j]]}, ignore_index=True)
    
    # save closest points to csv
    closest_points.to_csv('closest_points'+str(group)+'.csv')


    # Create a dictionary mapping cluster labels to bios
    #clusters_to_bios = {i: [] for i in range(5)}
    #for bio, label in zip(bios, labels):
        #clusters_to_bios[label].append(bio)

    # Now clusters_to_bios[i] is a list of all bios in cluster i
    #return clusters_to_bios

def save_clustered_bios(clusters_to_bios, group):
    with open('clustered_bios'+str(group)+'.pkl', 'wb') as f:
        pickle.dump(clusters_to_bios, f)


#save_clustered_bios(make_analysis(0,10), 0)
#save_clustered_bios(make_analysis(1,10), 1)
save_clustered_bios(make_analysis('Global',25), 'Global')




