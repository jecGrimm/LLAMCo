from prompt import load_ckp
import json

# ckp_file_1 = "output/Llama3_70B/test/0/answers_Llama3_70B_test_0_ckp_120.json"
# ckp_file_2 = "output/Llama3_70B/test/0/answers_Llama3_70B_test_0_ckp_580.json"

ckp_file_1 = "output/Llama3_70B/test/0/outputs_Llama3_70B_test_0_ckp_120.json"
ckp_file_2 = "output/Llama3_70B/test/0/outputs_Llama3_70B_test_0_ckp_580.json"

with open(ckp_file_1, "r", encoding = "utf-8") as f:
    #print("loading")
    ckp_data_1 = json.load(f)

with open(ckp_file_2, "r", encoding = "utf-8") as f:
    #print("loading")
    ckp_data_2 = json.load(f)
#ckp_data_1 = load_ckp(ckp_file_1)
#ckp_data_2 = load_ckp(ckp_file_2)

ckp_data = ckp_data_1 | ckp_data_2

# with open("output/Llama3_70B/test/0/answers_Llama3_70B_test_0_ckp.json", "w", encoding = "utf-8") as f:
#     json.dump(ckp_data, f, indent=4)

with open("output/Llama3_70B/test/0/outputs_Llama3_70B_test_0_ckp.json", "w", encoding = "utf-8") as f:
    json.dump(ckp_data, f, indent=4)