import pickle

#load clustered bios
with open('clustered_bios0.pkl', 'rb') as handle:
    clustered_bios = pickle.load(handle)

#print bios in cluster 0
for bio in clustered_bios[0]:
    print(bio)
    print('-----------------------------')
