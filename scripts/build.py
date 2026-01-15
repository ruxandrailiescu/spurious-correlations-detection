import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('./data/mmmu/m4.csv') 

subjects = sorted(df['label'].unique().tolist())
subject_to_idx = {s: i for i, s in enumerate(subjects)}

questions = df['text'].tolist()
labels = [subject_to_idx[l] for l in df['label']]

X_train, X_val, y_train, y_val = train_test_split(
    questions, labels, 
    test_size=0.20, 
    stratify=labels, 
    random_state=42
)

metadata = []
all_questions = []

for q, y in zip(X_train, y_train):
    metadata.append({'y': y, 'split': 0, 'a': 0})
    all_questions.append(q)

for q, y in zip(X_val, y_val):
    metadata.append({'y': y, 'split': 1, 'a': 0})
    all_questions.append(q)

metadata_df = pd.DataFrame(metadata)
questions_df = pd.DataFrame({'question_text': all_questions})

metadata_df.to_csv('./data/mmmu/metadata_mmmu.csv', index=False)
questions_df.to_csv('./data/mmmu/mmmu_questions.csv', index=False)

print(f"Total processed: {len(df)}")
print(f"Train size: {len(X_train)}")
print(f"Val size: {len(X_val)}")
print(f"Classes: {len(subjects)}")

for s in subjects:
    print(f"    '{s}',")