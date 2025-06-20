import json
from datasets import Dataset
import ast

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

if __name__=="__main__":
    read_output()