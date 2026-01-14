# from datasets import load_dataset
import pandas as pd
import re


# dataset = load_dataset('TIGER-Lab/MMLU-Pro')

# df = pd.read_csv('./data/mmlu_pro/mmlu_pro.csv')
# df = df[df['category'] == 'philosophy']
# df.to_csv('./data/mmlu_pro/mmlu_pro_philosophy.csv')


def parse_options(opt_str):
    pattern = r"'(.*?)'|\"(.*?)\""
    matches = re.findall(pattern, opt_str)
    return [m[0] if m[0] else m[1] for m in matches]


df = pd.read_csv('./data/mmlu_pro/mmlu_pro.csv')
processed_rows = []


for idx, row in df.iterrows():
    question = row['question']
    options_str = str(row['options'])
    correct_idx = int(row['answer_index'])
    
    options = parse_options(options_str)
    
    if not options or correct_idx >= len(options):
        continue
    
    correct_ans = options[correct_idx]
    processed_rows.append({
        'text': f"Question: {question}\nAnswer: {correct_ans}",
        'label': 'correct'
    })
    
    incorrect_idx = (correct_idx + 1) % len(options) 
    incorrect_ans = options[incorrect_idx]
    processed_rows.append({
        'text': f"Question: {question}\nAnswer: {incorrect_ans}",
        'label': 'incorrect'
    })


new_df = pd.DataFrame(processed_rows)
new_df.to_csv('./data/mmlu_pro/mmlu_pro_v2.csv', index=False)