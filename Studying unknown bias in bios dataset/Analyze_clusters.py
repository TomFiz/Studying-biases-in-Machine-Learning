import pickle
import pandas as pd

predictions = pd.read_csv('/var/tmp/tom.fizycki/Studying-biases-in-Machine-Learning-1/predicted_vs_true_jobGlobal.csv')
# check wether or not there are differences between the predicted and the true job
preds = predictions["predicted job"]
true = predictions["true job"]
diff = preds != true
print("Number of differences between the predicted and the true job:", sum(diff))

