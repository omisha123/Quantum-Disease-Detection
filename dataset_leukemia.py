import pandas as pd

# Load Kaggle files
train = pd.read_csv('data/data_set_ALL_AML_train.csv')
actual = pd.read_csv('data/actual.csv')

# Transpose genes to columns and merge labels
gene_cols = [c for c in train.columns if 'call' not in c and c not in ['Gene Description', 'Gene Accession Number']]
df = train[gene_cols].T
df.columns = train['Gene Accession Number']
df = df.reset_index().rename(columns={'index': 'patient'})
df['patient'] = df['patient'].astype(int)

# Merge target column
labels = actual.set_index('patient')['cancer'].map({'ALL': 0, 'AML': 1})
df['target'] = df['patient'].map(labels)
df = df.drop(columns=['patient']).dropna()

# Save formatted file for data_pipeline.py
df.to_csv('data/leukemia_gene_expression.csv', index=False)
print("Successfully generated data/leukemia_gene_expression.csv!")