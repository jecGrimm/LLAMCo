import json

def merge_ckpts():
    ckp_file_1 = "output/Llama3_70B/test/1/outputs_Llama3_70B_test_1_ckp.json"
    ckp_file_2 = "output/Llama3_70B/test/1/outputs_Llama3_70B_test_1_old.json"

    with open(ckp_file_1, "r", encoding = "utf-8") as f:
        ckp_data_1 = json.load(f)

    with open(ckp_file_2, "r", encoding = "utf-8") as f:
        #print("loading")
        ckp_data_2 = json.load(f)

    ckp_data = ckp_data_1 | ckp_data_2

    with open("output/Llama3_70B/test/1/outputs_Llama3_70B_test_1.json", "w", encoding = "utf-8") as f:
        json.dump(ckp_data, f, indent=4)

def count_model_dicts():
    model_out_file = f"output/Llama3_70B/test/1/outputs_Llama3_70B_test_1.json"

    with open(model_out_file, 'r', encoding="utf-8") as f:
        model_out = json.load(f)

    empt_counter = 0
    full_counter = 0
    for model_dict in model_out.values():
        if model_dict == "":
            empt_counter += 1
        else:
            full_counter += 1

    print("Empty rows: ", empt_counter)
    print("Full rows: ", full_counter)

if __name__ == "__main__":
    merge_ckpts()
    count_model_dicts()