# Biases-in-ML

This section studies the unknown biases in the bias in bios dataset.

To access the original dataset and presentation, check https://github.com/microsoft/biosbias.
The code is structured in 4 different steps :

1) Data preprocessing (BertMultiClf_1_DataPrep.py and BertMultiClf_1_DataPrep_neutral.py), the second one eliminating gender identification inside the bios.
2) Training (BertMultiClf_2_TrainTest.py and BertMultiClf_2_fit_NLP_model.py (called in the first one))
3) Testing and producing error dataset (BertMultiClf_3_CheckAll.py). To try processes on only a couple classes (medical sector for instance), go for BertMultiClf_3_NurseSurgeon.py
4) Analyzing biases in the error dataset (Error_Analysis.py, Analyze_clusters.py)

W2reg files are used for alternative LLMs fits, and were not used in our study.
