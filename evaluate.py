import json
from datasets import Dataset, load_from_disk
import ast
import os
from collections import defaultdict
import pandas as pd
from data import Data
import re

def read_output():
    with open("output/meta-llama/Llama-3.2-1B-Instruct/5/outputs_Llama-3.2-1B-Instruct_5.json", 'r', encoding="utf-8") as f:
        out = json.load(f)
    #print(out)

    pred_meta_list = []
    for item in out:
        model_out = item[0]["generated_text"][-1]["content"]
        
        try:
            dict_model_out = ast.literal_eval(clean_model_out(model_out))
            
            pred_meta_list.append(dict_model_out)
        except:
            pass

    # [ast.literal_eval(item[0]["generated_text"][-1]["content"]) for item in out]
    pred_meta = Dataset.from_list(pred_meta_list)
    print(len(pred_meta))

def clean_model_out(model_out: str):
    cleaned_first = '{'+model_out.split("{")[-1]
    cleaned = cleaned_first.split("}")[0]+'}' 
    return cleaned

def read_wiki_out():
    author_wiki_out = load_from_disk("output/wikidata/hf/author")
    print(len(author_wiki_out))

def evaluate_wiki():
    data = Data()
    eval_samples = data.eval_samples

    author_wiki_out = load_from_disk("output/wikidata/hf/author")
    work_wiki_out = load_from_disk("output/wikidata/hf/work")

    cols = [feat for feat in eval_samples.features if feat != "Dokument_ID"]
    eval_out = eval_samples.map(lambda x: validate_wiki_sample(sample = x, author_wiki_data=author_wiki_out, work_wiki_data = work_wiki_out, cols = cols))

    # accuracy
    total_acc = 0.0
    row_acc = defaultdict(float)
    column_acc = {col: 0.0 for col in cols}
    num_cols = len(cols)
    num_rows = len(eval_out)

    for i, val_lbls in enumerate(eval_out["Label"]):
        corr_lbls = sum(val_lbls)
        total_acc += corr_lbls # count correct labels
        row_acc[eval_out["Dokument_ID"][i]] = corr_lbls/num_cols

        for j, col in enumerate(cols): # TODO: ist das dieselbe Reihenfolge wie bei val_lbls?
            if val_lbls[j]:
                column_acc[col] += 1
    
    total_acc = total_acc/(num_rows*num_cols)

    column_acc = {col:(corr/num_rows) for col, corr in column_acc.items()}

    max_row_idx = max(row_acc, key = lambda x: row_acc[x])
    min_row_idx = min(row_acc, key = lambda x: row_acc[x])

    max_col_idx = max(column_acc, key = lambda x: column_acc[x])
    min_col_idx = min(column_acc, key = lambda x: column_acc[x])

    # TODO: format as 0.00
    output = "Evaluation:\n"
    output += f"\nTotal Accuracy: {total_acc}\n\n"
    output += f"Row with the highest Accuracy: {max_row_idx} - {row_acc[max_row_idx]}\n"
    output += f"Row with the highest Accuracy: {min_row_idx} - {row_acc[min_row_idx]}\n"
    
    output += f"\nColumn with the highest Accuracy: {max_col_idx} - {column_acc[max_col_idx]}\n"
    output += f"Column with the highest Accuracy: {min_col_idx} - {column_acc[min_col_idx]}\n"

    output += f"\nAccuracy per column:\n"
    for col, acc in column_acc.items():
        output += f"{col}: {acc}\n"
    
    with open(f"output/wikidata/evaluation_wiki.txt", 'w',encoding = "utf-8") as f:
        f.write(output)

def validate_wiki_sample(sample, author_wiki_data, work_wiki_data, cols):
    val_labels = []
    author_wiki_sample = author_wiki_data.filter(lambda x: x["Dokument_ID"] == sample["Dokument_ID"]) 
    work_wiki_sample = work_wiki_data.filter(lambda x: x["Dokument_ID"] == sample["Dokument_ID"]) 
    
    wiki_vals = set(author_wiki_sample["Output"][0]).union(set(work_wiki_sample["Output"][0]))
        
    # get years from datestrings
    for val in author_wiki_sample["Output"][0]:
        match = re.match(r"(\d{4})-.*Z", val)
        if match:
            wiki_vals.add(match.group(1))

    for val in work_wiki_sample["Output"][0]:
        match = re.match(r"(\d{4})-.*Z", val)
        if match:
            wiki_vals.add(match.group(1))

    prepped_sample = prep_eval_data(sample)
    # eval_vals = set(prepped_sample.values())

    # found_vals = eval_vals & wiki_vals

    if len(wiki_vals) != 0: 
        for col in cols:
            if prepped_sample[col] in wiki_vals or str(prepped_sample[col]).lower() in wiki_vals:
                val_labels.append(True)
            else:
                val_labels.append(False)
    else:
        val_labels = [False for _ in range(len(cols))]

    return {f"Label": val_labels}

def evaluate_llama(experiment_mode = "dev", model_id = "Llama3_70B"):
    output = "Evaluation:\n"

    data = Data()

    path = f"./output/{model_id}/{experiment_mode}"
    num_cols = len(data.eval_samples.features) - 1

    shot_dirs = [dir for dir in os.listdir(path) if dir.isdigit()]
    for shot in shot_dirs:
        output = "Evaluation:\n"
        eval_samples = data.eval_samples
        shot = int(shot)
        model_out_file = f"{path}/{shot}/outputs_{model_id}_{experiment_mode}_{shot}.json"

        with open(model_out_file, 'r', encoding="utf-8") as f:
            model_out = json.load(f)

        # check correct labels
        eval_samples = eval_samples.skip(shot)        

        if experiment_mode == "dev":
            eval_samples = eval_samples.select(range(10))
        else:
            eval_samples = eval_samples.skip(10)
        
        num_rows = len(eval_samples) # exclude dev samples and shots 

        # TODO: Fix bool-return bei shot = 5
        eval_out = eval_samples
        #eval_out = eval_samples.map(lambda x: prep_eval_data(x))
        cols = [feat for feat in eval_out.features if feat not in ["Dokument_ID", "Label"]]
        eval_out = eval_out.map(lambda x: validate_sample(x, model_out, cols = cols))    
        
        # accuracy
        total_acc = 0.0
        row_acc = defaultdict(float)
        column_acc = {col: 0.0 for col in cols}

        for i, val_lbls in enumerate(eval_out["Label"]):
            corr_lbls = sum(val_lbls)
            total_acc += corr_lbls # count correct labels
            row_acc[eval_out["Dokument_ID"][i]] = corr_lbls/num_cols

            for j, col in enumerate(cols): # TODO: ist das dieselbe Reihenfolge wie bei val_lbls?
                if val_lbls[j]:
                    column_acc[col] += 1
        
        total_acc = total_acc/(num_rows*num_cols)

        column_acc = {col:(corr/num_rows) for col, corr in column_acc.items()}

        max_row_idx = max(row_acc, key = lambda x: row_acc[x])
        min_row_idx = min(row_acc, key = lambda x: row_acc[x])

        max_col_idx = max(column_acc, key = lambda x: column_acc[x])
        min_col_idx = min(column_acc, key = lambda x: column_acc[x])

        output += f"\n{shot}-shot:\n\n"
        output += f"Total Accuracy: {total_acc}\n\n"
        output += f"Row with the highest Accuracy: {max_row_idx} - {row_acc[max_row_idx]}\n"
        output += f"Row with the highest Accuracy: {min_row_idx} - {row_acc[min_row_idx]}\n"
        
        output += f"\nColumn with the highest Accuracy: {max_col_idx} - {column_acc[max_col_idx]}\n"
        output += f"Column with the highest Accuracy: {min_col_idx} - {column_acc[min_col_idx]}\n"

        output += f"\nAccuracy per column:\n"
        for col, acc in column_acc.items():
            output += f"{col}: {acc}\n"
        
        with open(f"{path}/{shot}/evaluation_{model_id}_{experiment_mode}_{shot}.txt", 'w',encoding = "utf-8") as f:
            f.write(output)

def validate_sample(sample, model_out, cols):
    model_dict = model_out[sample["Dokument_ID"]]
    num_cols = len(cols)

    val_labels = []
    if model_dict != "":
        prepped_sample = prep_eval_data(sample)
        for col in cols:
            model_val = model_dict[col]

            # if model_val != "" and type(model_val) == str and type(prepped_sample[col]) == int:
            #     model_val = int(model_val)

            if (prepped_sample[col] == model_val) or (prepped_sample[col] in [None, "unknown"] and model_val == '') or (str(prepped_sample[col]).lower().strip() == str(model_val).lower().strip()):
                val_labels.append(True)
            else:
                val_labels.append(False) #TODO: check for Kanon_Status and Jahr_ED
            #val_labels.append((sample[col] == model_val))
    else:
        val_labels = [False for _ in range(num_cols)]
    
    return {"Label": val_labels}

def prep_eval_data(sample):
    prepped_sample = sample
    for col, val in sample.items():
        if val == None:
            prepped_sample[col] = ""
        else:
            if col in ["Jahr_ED", "entstanden", "Jahr_Zweitdruck", "Jahr_Drittdruck", "Kanon_Status"]:
                try:
                    prepped_sample[col] = str(int(val))
                except:
                    prepped_sample[col] = str(val)
            elif col in ["seriell", "in_Deutscher_Novellenschatz_(Heyse)", "in_Pantheon", "in_B-v-Wiese"]:
                if val.lower() in ["1", "true"]:
                    prepped_sample[col] = "true"
                else:
                    prepped_sample[col] = "false"
               # sample[col] = bool(val)
    return prepped_sample

if __name__=="__main__":
    #evaluate_wiki()
    evaluate_llama()