import json
from datasets import load_from_disk
import os
from collections import defaultdict
import pandas as pd
from data import Data
import re
import argparse
import matplotlib.pyplot as plt

def evaluate_wiki():
    """
    This function evaluates the information retrieved by wikidata.
    """
    data = Data()
    eval_samples = data.eval_samples

    author_wiki_out = load_from_disk("output/wikidata/hf/author")
    work_wiki_out = load_from_disk("output/wikidata/hf/work")

    # exclude columns that cannot be contained in wikidata
    cols = [feat for feat in eval_samples.features if feat not in ["seriell", "in_Deutscher_Novellenschatz_(Heyse)", "in_Pantheon", "in_B-v-Wiese", "in_RUB_Sammlung", "Kanon_Status", "Dokument_ID"]]
    total_metrics, row_metrics, col_metrics = validate_wiki_samples(eval_samples, author_wiki_out, work_wiki_out, cols)

    outfile = f"output/wikidata/evaluation_wiki"
    json.dump(total_metrics, open(outfile+"_total.json", 'w', encoding="utf-8"), indent=4)
    json.dump(row_metrics, open(outfile+"_rows.json", 'w', encoding="utf-8"),  indent=4)
    json.dump(col_metrics, open(outfile+"_cols.json", 'w', encoding="utf-8"),  indent=4)

def validate_wiki_samples(eval_samples, author_wiki_data, work_wiki_data, cols: list):
    """
    This function compares the wikidata samples with the corpus samples.

    @params
        eval_samples: evaluation corpus
        author_wiki_data: information about the author retrieved from wikidata
        work_wiki_data: information about the publication retrieved from wikidata
        cols: evaluated columns
    @returns
        total_metrics: metrics for all cells in the corpus
        row_metrics: metrics per row in the corpus
        col_metrics: metrics per column in the corpus
    """
    total_tp = 0.0
    total_fn = 0.0

    row_tp = defaultdict(float)
    row_fn = defaultdict(float)

    col_tp = {col: 0.0 for col in cols}
    col_fn = {col: 0.0 for col in cols}

    for sample in eval_samples:
        idx = sample["Dokument_ID"]
        author_wiki_sample = author_wiki_data.filter(lambda x: x["Dokument_ID"] == idx) 
        work_wiki_sample = work_wiki_data.filter(lambda x: x["Dokument_ID"] == idx) 
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
        for col in cols:
            if prepped_sample[col] not in [None, "unknown", ""]: # only compute recall
                if prepped_sample[col] in wiki_vals or str(prepped_sample[col]).lower() in wiki_vals:
                    total_tp += 1
                    row_tp[idx] += 1
                    col_tp[col] += 1
                else:
                    total_fn += 1
                    row_fn[idx] += 1
                    col_fn[col] += 1
    
    if total_tp != 0 or total_fn != 0:
        total_metrics = {
            "recall": total_tp/(total_tp + total_fn), 
            "true_positive": total_tp, 
            "true_negative": 0.0,
            "false_positive": 0.0,
            "false_negative": total_fn,
            "amount": total_tp + total_fn
        }
    else:
        total_metrics = {
            "recall": 0.0,
            "true_positive": total_tp, 
            "true_negative": 0.0,
            "false_positive": 0.0,
            "false_negative": total_fn,
            "amount": total_tp + total_fn
        }

    row_metrics = defaultdict(dict)
    for idx in eval_samples["Dokument_ID"]:
        if row_tp[idx] != 0 or row_fn[idx] != 0:
            row_metrics[idx] = {
                "recall": row_tp[idx]/(row_tp[idx] + row_fn[idx]),
                "true_positive": row_tp[idx], 
                "true_negative": 0.0,
                "false_positive": 0.0,
                "false_negative": row_fn[idx],
                "amount": row_tp[idx] + row_fn[idx]
            }
        else: 
            row_metrics = {
                "recall": 0.0,
                "true_positive": row_tp[idx], 
                "true_negative": 0.0,
                "false_positive": 0.0,
                "false_negative": row_fn[idx],
                "amount": row_tp[idx] + row_fn[idx]
            }

    col_metrics = defaultdict(dict)
    for col in cols:
        if col_tp[col] != 0 or col_fn[col] != 0:
            col_metrics[col] = {
                "recall": col_tp[col]/(col_tp[col] + col_fn[col]),
                "true_positive": col_tp[col], 
                "true_negative": 0.0,
                "false_positive": 0.0,
                "false_negative": col_fn[col],
                "amount": col_tp[col] + col_fn[col]
            }
        else: 
            col_metrics = {
                "recall": 0.0,
                "true_positive": col_tp[col], 
                "true_negative": 0.0,
                "false_positive": 0.0,
                "false_negative": col_fn[col],
                "amount": col_tp[col] + col_fn[col]
            }
    
    return total_metrics, row_metrics, col_metrics

def evaluate_llm(experiment_mode = "dev", model_id = "Llama3_70B"):
    """
    This function evaluates the outputs of the Llama models.

    @params
        experiment_mode: evaluate on dev or test split
        model_id: model to evaluate
    """
    data = Data()

    path = f"./output/{model_id}/{experiment_mode}"

    shot_dirs = [dir for dir in os.listdir(path) if dir.isdigit()]
    for shot in shot_dirs:
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
        
        cols = [feat for feat in eval_samples.features if feat not in ["Dokument_ID", "Label"]]

        total_metrics, row_metrics, col_metrics = validate_samples(eval_samples, model_out, cols)

        outfile = f"{path}/{shot}/evaluation_{model_id}_{experiment_mode}_{shot}"
        json.dump(total_metrics, open(outfile+"_total.json", 'w', encoding="utf-8"), indent=4)
        json.dump(row_metrics, open(outfile+"_rows.json", 'w', encoding="utf-8"),  indent=4)
        json.dump(col_metrics, open(outfile+"_cols.json", 'w', encoding="utf-8"),  indent=4)

def validate_samples(eval_samples, model_out: dict, cols: list):
    """
    This function counts the correctly generated values.

    @params
        eval_samples: correct sample values
        model_out: generated values
        cols: column names
    @returns
        total_metrics: dictionary with the accuracy, precision, recall, and f1 for all cells
        row_metrics: dictionary with the accuracy, precision, recall, and f1 per instance
        col_metrics: dictionary with the accuracy, precision, recall, and f1 per column 
    """
    total_tp = 0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0

    row_tp = defaultdict(float)
    row_tn = defaultdict(float)
    row_fp = defaultdict(float)
    row_fn = defaultdict(float)

    col_tp = {col: 0.0 for col in cols}
    col_tn = {col: 0.0 for col in cols}
    col_fp = {col: 0.0 for col in cols}
    col_fn = {col: 0.0 for col in cols}

    for sample in eval_samples:
        idx = sample["Dokument_ID"]

        if idx in model_out.keys():
            model_dict = model_out[idx]

            prepped_sample = prep_eval_data(sample)
            if model_dict != "":
                for col in cols:
                    model_val = model_dict[col]

                    if prepped_sample[col] not in [None, "unknown", "", "o.N.", "(unbekannt)"]:
                        if prepped_sample[col] == model_val or (str(prepped_sample[col]).lower().strip() == str(model_val).lower().strip()):
                            total_tp += 1
                            row_tp[idx] += 1
                            col_tp[col] += 1
                        elif col == "Kanon_Status" and str(model_val).lower().strip() in ["kanonisch", "kanonisiert", "kanon", "canon"] and str(prepped_sample[col]).lower().strip() in ["2", "3"]:
                            # TODO: schauen wegen nicht kanonisch
                            total_tp += 1
                            row_tp[idx] += 1
                            col_tp[col] += 1
                        else:
                            total_fn += 1
                            row_fn[idx] += 1
                            col_fn[col] += 1
                    else:
                        if model_val not in [None, "unknown", "",  "o.N.", "(unbekannt)"]:
                            total_fp += 1
                            row_fp[idx] += 1
                            col_fp[col] += 1
                        else:
                            total_tn += 1
                            row_tn[idx] += 1
                            col_tn[col] += 1
    
    total_metrics = defaultdict(float)
    total_metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
    total_metrics["true_positive"] = total_tp
    total_metrics["true_negative"] = total_tn
    total_metrics["false_positive"] = total_fp
    total_metrics["false_negative"] = total_fn
    total_metrics["amount"] = total_tp + total_tn + total_fp + total_fn

    row_metrics = defaultdict(dict)
    for idx in eval_samples["Dokument_ID"]:
        row_metrics[idx] = calculate_metrics(row_tp[idx], row_tn[idx], row_fp[idx], row_fn[idx])
        row_metrics[idx]["true_positive"] = row_tp[idx]
        row_metrics[idx]["true_negative"] = row_tn[idx]
        row_metrics[idx]["false_positive"] = row_fp[idx]
        row_metrics[idx]["false_negative"] = row_fn[idx] 
        row_metrics[idx]["amount"] = row_tp[idx] + row_tn[idx] + row_fp[idx] + row_fn[idx]

    col_metrics = defaultdict(dict)
    for col in cols:
        col_metrics[col] = calculate_metrics(col_tp[col], col_tn[col], col_fp[col], col_fn[col])
        col_metrics[col]["true_positive"] = col_tp[col]
        col_metrics[col]["true_negative"] = col_tn[col]
        col_metrics[col]["false_positive"] = col_fp[col]
        col_metrics[col]["false_negative"] = col_fn[col] 
        col_metrics[col]["amount"] = col_tp[col] + col_tn[col] + col_fp[col] + col_fn[col]
    
    return total_metrics, row_metrics, col_metrics

def calculate_metrics(tp, tn, fp, fn):
    """
    This function calculates the evaluation metrics.

    @params 
        tp: true positives
        tn: true negatives
        fp: false positives
        fn: false negatives
    @returns dictionary containing the accuracy, precision, recall, and f1
    """
    if tp != 0 or fp != 0:
        prec = tp/(tp + fp)
    else:
        prec = 0.0
    
    if tp != 0 or fn != 0:
        rec = tp/(tp + fn)
    else:
        rec = 0.0

    if prec != 0 or rec != 0:
        f1 = (2*prec*rec)/(prec + rec)
    else:
        f1 = 0.0

    if tp != 0 or fn != 0 or fp != 0 or tn != 0: # empty model_dict
        acc = ((tp + tn)/(tp + tn + fp + fn))
    else:
        acc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def prep_eval_data(sample):
    """
    This function preprocesses the evaluation samples for the comparison with the model.

    @params
        sample: evaluation sample
    @returns
        prepped_sample: preprocessed evaluation sample
    """
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
            elif col in ["seriell", "in_Deutscher_Novellenschatz_(Heyse)", "in_Pantheon", "in_B-v-Wiese", "in_RUB_Sammlung"]:
                if str(val).lower() in ["1", "true"]:
                    prepped_sample[col] = "true"
                else:
                    prepped_sample[col] = "false"
    return prepped_sample

def plot_cols(experiment_mode = "test", model_id = "Llama3_8B"):
    """
    This function creates a plot for the column metrics.

    @params
        experiment_mode: evaluate on dev or test split
        model_id: model to evaluate 
    """
    path = f"./output/{model_id}/{experiment_mode}"

    shot_dirs = [dir for dir in os.listdir(path) if dir.isdigit()]
    col_recall = dict()
    for shot in shot_dirs:
        with open(f"{path}/{shot}/evaluation_{model_id}_{experiment_mode}_{shot}_cols.json", "r", encoding = "utf-8") as f:
            col_metrics = json.load(f)
        
        col_recall[shot] = [metrics["recall"] for metrics in col_metrics.values()]
    
    df = pd.DataFrame(data = col_recall, index = col_metrics.keys(), columns = ["0", "1", "5"])
    df.plot.bar(ylabel = "Recall", xlabel = "Metadata category", title = f"Recall per metadata category: {model_id.replace("_", "-")}")
    plt.legend(title = "Shots", loc=(0.3, 0.48), fontsize = "x-small", title_fontsize = "x-small")
    plt.tight_layout()
    plt.savefig(f"{path}/evaluation_{model_id}_{experiment_mode}_cols.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', help='model to run the experiment with', default = "Llama3_8B")
    parser.add_argument('--experiment_mode', '-e', help='experiment mode', default = "test")
    model_name = parser.parse_args().model_name
    experiment_mode = parser.parse_args().experiment_mode

    #evaluate_wiki()
    evaluate_llm(model_id=model_name, experiment_mode=experiment_mode)
    plot_cols(model_id=model_name, experiment_mode=experiment_mode)