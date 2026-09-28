import os
import pandas as pd

# Kaggle crawford/gene-expression files, placed in data/
TRAIN_PATH = 'data/data_set_ALL_AML_train.csv'
TEST_PATH = 'data/data_set_ALL_AML_independent.csv'
LABELS_PATH = 'data/actual.csv'
OUT_PATH = 'data/leukemia_gene_expression.csv'


def load_expression(path):
    """Read one Golub file and return a patients x genes DataFrame (index = patient id)."""
    raw = pd.read_csv(path)
    # Drop the 'call' columns (call, call.1, ...) and the two description columns
    patient_cols = [c for c in raw.columns
                    if not c.startswith('call') and c not in ['Gene Description', 'Gene Accession Number']]
    expr = raw[patient_cols].T
    expr.columns = raw['Gene Accession Number']
    expr.index = expr.index.astype(int)
    expr.index.name = 'patient'
    return expr


train = load_expression(TRAIN_PATH)
frames = [train]

if os.path.exists(TEST_PATH):
    test = load_expression(TEST_PATH)
    # Keep only genes present in both files, in the same order
    common = train.columns.intersection(test.columns)
    frames = [train[common], test[common]]
    print(f"Merging train ({len(train)}) + independent ({len(test)}) patients")
else:
    print(f"WARNING: {TEST_PATH} not found - using train file only ({len(train)} patients)")

df = pd.concat(frames)
# Some accession numbers are duplicated in the original file; keep the first occurrence
df = df.loc[:, ~df.columns.duplicated()]

# Merge target column
actual = pd.read_csv(LABELS_PATH)
labels = actual.set_index('patient')['cancer'].map({'ALL': 0, 'AML': 1})
df['target'] = df.index.map(labels)
df = df.dropna(subset=['target'])
df['target'] = df['target'].astype(int)
df = df.sort_index().reset_index(drop=True)

# Save formatted file for export_results.py
df.to_csv(OUT_PATH, index=False)
print(f"Successfully generated {OUT_PATH}: {len(df)} patients, "
      f"{df.shape[1] - 1} genes, class counts {df['target'].value_counts().to_dict()}")