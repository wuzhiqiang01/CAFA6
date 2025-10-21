import pandas as pd


df = pd.read_csv(r"C:\Users\17539\Desktop\cafa\cafa-6-protein-function-prediction\sample_submission.tsv")

# df.columns = ["The Protein ID", "The Gene Ontology term (GO) ID", "Predicted link probability that GO appear in Protein"]
print(df.head())