import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from transformers import DistilBertModel, DistilBertTokenizer

# Load the pkl file
def make_analysis(group,n_clusters):

    with open('Treated_Error'+str(group)+'.pkl', mode='rb') as f:
        error = pickle.load(f)
        X = error['X']
        y = error['y']
        bios = error['bio']
        #y_pred = error['predicted_job']

    # plot number of errors per job
    nbr_errors_true = np.sum(y, axis = 0) 
    plt.figure()
    plt.bar(np.arange(len(nbr_errors_true)), nbr_errors_true)
    plt.xlabel('True Job')
    plt.ylabel('Number of errors')
    plt.title('Number of errors per true job for group ' + str(group))
    plt.savefig('errors_per_job'+str(group)+'.pdf')

    # plot number of errors per predicted job
    #nbr_errors_pred = np.sum(y_pred, axis = 0)
    #plt.figure()
    #plt.bar(np.arange(len(nbr_errors_pred)), nbr_errors_pred)
    #plt.xlabel('Predicted Job')
    #plt.ylabel('Number of errors')
    #plt.title('Number of errors per predicted job for group ' + str(group))
    #plt.savefig('errors_per_predicted_job'+str(group)+'.pdf')


    # Load the DistilBert model and tokenizer
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)

    def get_embeddings(list_input_texts, tokenizer, Max_len):
        embeddings = []
        for input_text in list_input_texts:
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
            cls_output = output[0][0][0].numpy()
            embeddings.append(cls_output)

        return embeddings

    # Get embeddings for all input sentences
    embeddings = get_embeddings(bios, tokenizer, Max_len=512)

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters)  # Use 5 clusters as an example
    kmeans.fit(embeddings)
    
    # Get the cluster labels for each bio
    labels = kmeans.labels_

    # Create a dictionary mapping cluster labels to bios
    clusters_to_bios = {i: [] for i in range(5)}
    for bio, label in zip(bios, labels):
        clusters_to_bios[label].append(bio)

    # Now clusters_to_bios[i] is a list of all bios in cluster i
    return clusters_to_bios

def save_clustered_bios(clusters_to_bios, group):
    with open('clustered_bios'+str(group)+'.pkl', 'wb') as f:
        pickle.dump(clusters_to_bios, f)

save_clustered_bios(make_analysis(0,5), 0)
save_clustered_bios(make_analysis(1,5), 1)




