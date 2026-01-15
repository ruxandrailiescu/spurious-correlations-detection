from datasets import load_dataset
import pandas as pd


cat = ['Art_Theory', 'Basic_Medical_Science', 'Biology', 
       'Chemistry', 'Clinical_Medicine', 'Computer_Science',
       'Design', 'Diagnostics_and_Laboratory_Medicine',
       'Economics', 'Electronics', 'Energy_and_Power',
       'Finance', 'Geography', 'History', 'Literature',
       'Manage', 'Marketing', 'Materials', 'Mechanical_Engineering',
       'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health',
       'Sociology']
df = pd.read_csv('./data/m4.csv')

for c in cat:
    ds = load_dataset('MMMU/MMMU', c)
    a = pd.concat([ds['dev'].to_pandas(),ds['validation'].to_pandas(),ds['test'].to_pandas()],ignore_index=True)
    a = pd.DataFrame({
        'text':a['question'],
        'label':a['topic_difficulty']
    })
    df = pd.concat([df, a], ignore_index=True)

df.to_csv('./data/m4.csv', index=False)