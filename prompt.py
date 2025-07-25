import torch
from transformers import pipeline
import os
from data import Data
from tqdm.auto import tqdm
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import ast
from collections import defaultdict
import argparse

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\n\
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n\
Please ensure that your responses are socially unbiased and positive in nature.\n\
\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n\
If you don’t know the answer to a question, please don’t share false information.\n\
If you don’t know the answer to a question, please answer with "NAN".\n\
"""

DEFAULT_SYSTEM_PROMPT_DE = """\
Du bist ein hilfreicher, respektvoller und ehrlicher Assistent. Antworte immer so hilfreich wie möglich und bleibe dabei ein sicherer Umgang.\n\
Deine Antworten sollten keinen verletzenden, unethischen, rassistischen, sexistischen, toxischen, gefährlichen oder illegalen Inhalt enthalten.\n\
\n\
"""

DEFAULT_PROMPT = """\
Please help me to collect structured metadata information about a published literary text.\n\
Create a dictionary containing only the given output keys and the corresponding missing information for the author and title given by the input keys.\n\
Please refer to the explanations of the output keys to fill in the correct information.\n\
Please return "" as value if you don't know the answer.\n\
Please return only the output dictionary.\n\
Please answer in German.\n\
\n\
Input keys:\n\
"author": name of an author\n\
"title": name of a creative work written by the author\n\
\n\
Output keys:\n\
"last_name": last name of the author given in the input\n\
"first_name": first names of the author given in the input\n\
"pseudonym": pseudonym of the author given in the input\n\
"gender": assumed gender of the author given in the input; possible values "m" (male), "f" (female)\n\
"title": title of the creative work given in the input\n\
"subtitle in the text": subtitle in the printed text of the creative work given in the input\n\
"subtitle in the table of contents": subtitle in the table of contents of the creative work given in the input\n\
"year_first_publication": year of the first publication of the creative work given in the input\n\
"developed": year of the development of the creative work given in the input\n\
"genre_first_publication": genre of the first publication of the creative work given in the input\n\
"medium_first_publication": medium of the first publication of the creative work given in the input\n\
"medium_type_first_publication": type of the medium of the first publication of the creative work given in the input\n\
"publisher": publisher of the first publication of the creative work given in the input\n\
"canon_status": canon status of the author given in the input; possible values are 0 (unknown author, author not part of the literary canon, no digitalized texts available on the internet), 1 (author unknown today, author not part of the literary canon, digitalized texts available on the internet), 2 (author known, author not part of the literary canon, digitalized texts available on the internet), 3 (author known, author part of the literary canon, digitalized texts available on the internet)\n\
"serial": if the creative work given in the input was published serially; possible values are True (creative work given in the input was published serially), False (creative work given in the input was not published serially)\n\
"pages": page numbers of the first publication of the creative work given in the input; format of the values is <first_page_number>-<last_page_number>\n\
"medium_second_publication": medium of the second publication of the creative work given in the input\n\
"year_second_publication": year of the second publication of the creative work given in the input\n\
"genre_second_publication": genre of the second publication of the creative work given in the input\n\
"medium_third_publication": medium of the third publication of the creative work given in the input\n\
"year_third_publication": year of the third publication of the creative work given in the input\n\
"genre_third_publication": genre of the third publication of the creative work given in the input\n\
"in_Deutscher_Novellenschatz_Heyse": if the creative work given in the input is listed in "Deutscher Novellenschatz" by Paul Heyse; possible values are True (creative work given in the input is listed in "Deutscher Novellenschatz" by Paul Heyse), False (creative work given in the input is not listed in "Deutscher Novellenschatz" by Paul Heyse)\n\
"in_Pantheon_Hoffmann": if the creative work given in the input is listed in "Pantheon" by Carl Hoffmann (publisher); possible values are True (creative work given in the input is listed in "Pantheon" by Carl Hoffmann (publisher)), False (creative work given in the input is not listed in "Pantheon" by Carl Hoffmann (publisher))\n\
"in_Novelle_von_Wiese": if the creative work given in the input is listed in "Novelle" by Benno von Wiese; possible values are True (creative work given in the input is listed in "Novelle" by Benno von Wiese), False (creative work given in the input is not listed in "Novelle" by Benno von Wiese)\n\
\n\
"""

DEFAULT_PROMPT_DE = """Du bist ein wissenschaftlicher Assistent und hast die Aufgabe, bibliografische und literaturwissenschaftliche Metadaten zu einem literarischen Werk zu erheben.

Gegeben ist ein Input mit den drei Informationen:
- Vorname des Autors  
- Nachname des Autors  
- Titel des literarischen Textes  

Erstelle ein Python-Dictionary im vorgegebenen Format. Das Dictionary soll nur die unten aufgeführten **Output Keys** enthalten. Trage zu jedem Key die passende Information ein. Nutze dabei dein vorhandenes Wissen oder führe eine Websuche durch.

Wichtige Hinweise:
- Gib ausschließlich das ausgefüllte Dictionary im genannten Format zurück – keine Kommentare, keine zusätzliche Erklärung, kein Fließtext.  
- Antworte ausschließlich auf Deutsch.  
- Gib bei "Gender" entweder "m" oder "f" zurück.
- Gib bei "seriell", "in_Deutscher_Novellenschatz_(Heyse)", "in_Pantheon", "in_RUB_Sammlung" und "in_B-v-Wiese" entweder "true" oder "false" zurück.
- Gib das Ergebnis exakt im folgenden Format zurück:

{
    "Vorname": "",
    "Nachname": "",
    "Pseudonym": "",
    "Gender": "",
    "Titel": "",
    "Untertitel_im_Text": "",
    "Untertitel_im_Inhaltsverzeichnis": "",
    "Jahr_ED": "",
    "entstanden": "",
    "Gattungslabel_ED": "",
    "Medium_ED": "",
    "Medientyp_ED": "",
    "Hg.": "",
    "Kanon_Status": "",
    "seriell": "",
    "Seiten": "",
    "Medium_Zweitdruck": "",
    "Jahr_Zweitdruck": "",
    "Label_Zweitdruck": "",
    "Medium_Drittdruck": "",
    "Jahr_Drittdruck": "",
    "Label_Drittdruck": "",
    "in_Deutscher_Novellenschatz_(Heyse)": "",
    "in_Pantheon": "",
    "in_B-v-Wiese": "",
    "in_RUB_Sammlung": ""
}

Hinweise: "ED" steht für "Erstdruck", "Hg." bezieht sich auf den Herausgeber der Anthologie oder Almanachs etc., in dem der Text ggf. erschienen ist. "in_Pantheon" bedeutet, dass der Text in der Novellensammlung "Pantheon" erschienen ist. "in_B-v-Wiese" bedeutet, dass der Text in "Benno von Wiese: Die deutsche Novelle von Goethe bis Kafka. Interpretationen" erschienen ist. "In_RUB_Sammlung" bedeutet, dass der Text in "Erzählungen und Novellen des 19. Jahrhunderts" im Reclam-Verlag erschienen ist.
"""

class Prompter:
    def __init__(self, model_id, experiment_mode):
        self.model_id = model_id
        self.experiment_mode = experiment_mode

    def create_messages(self, prompt, system_prompt = DEFAULT_SYSTEM_PROMPT_DE):
        """
        This function creates the prompt template for Llama.

        @params
            prompt: x-shot prompt
            system_prompt: system prompt
        @returns messages: x-shot and system prompt combined in the llama chat template
        """
        messages= [\
            {"role": "system", "content": system_prompt},\
            {"role": "user", "content": prompt},\
        ]
        return messages

    def create_few_shot_samples(self, eval_dataset, shots = 0):
        """
        This function turns the few-shot samples into a prompting format.

        @params
            eval_dataset: Huggingface dataset with the evaluation samples
            shots: number of x-shot samples
        @returns few_shots: x-shot samples in prompt format 
        """
        few_shots = ""
        for shot in range(shots):
            few_shots += f"Input: {{'Vorname': '{eval_dataset["Vorname"][shot]}, 'Nachname': '{eval_dataset["Nachname"][shot]}, 'Titel': {eval_dataset["Titel"][shot]}}}\n"
            few_shots += f"Output: {eval_dataset.select([shot])}\n"
        return few_shots

    def create_prompt(self, prompt_sample, few_shots, instructions=DEFAULT_PROMPT_DE):
        """
        This function combines the parts of the prompt.

        @params
            prompt_sample: current sample 
            few_shots: x-shot samples
            instructions: prompt instructions
        @returns full prompt consisting of the instruction, x examples and the current sample
        """
        return instructions + few_shots + f"Input: {prompt_sample}\nOutput: "

    def generate_text(self, sample, pipe, few_shot, max_new_tokens = 500, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
        """
        This function creates the prompts the model and returns the output text.

        @params
            sample: current metadata sample
            pipe: text generation pipeline
            few_shot: x-shot examples
            max_new_tokens: token generation limit
            instructions: prompt instructions
            system_prompt: system prompt instructions 
        @returns dictionary containing the output text
        """
        prompt = self.create_prompt(sample, few_shots=few_shot, instructions=instructions)
        messages = self.create_messages(prompt, system_prompt=system_prompt)

        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
        )
        return {"outputs": outputs[0]["generated_text"][-1]} # dictionary to turn it automatically into a Huggingface dataset

    def create_message_dataset(self, sample, few_shot, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
        prompt = self.create_prompt(sample, few_shots=few_shot, instructions=instructions)
        messages = self.create_messages(prompt, system_prompt=system_prompt)
        return {"messages": messages}

    def prompt_model(self, prompt_dataset, eval_dataset, model_id = "meta-llama/Llama-3.2-1B-Instruct", shots = 0, max_new_tokens = 500, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
        """
        @deprected: Use prompt_model_dataset instead
        This function prompts the whole dataset to the model in a x-shot manner. Saves the generated text as a Huggingface dataset (not human readable) and as a json file (human readable).

        @params
            prompt_dataset: prompt samples
            eval_dataset: evaluation samples
            model_id: Huggingface model name
            shots: number of examples in the instructions
            max_new_tokens: limit of the number of generated tokens
            instructions: model instructions
            system_prompt: general system instructions 
        """
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        few_shots = self.create_few_shot_samples(eval_dataset, shots)
        
        prompt_dataset = prompt_dataset.select([i for i in range(shots, len(prompt_dataset))]) # Remove few-shot-examples
        
        outputs = prompt_dataset.map(lambda x: self.generate_text(sample=x, pipe=pipe, few_shot=few_shots, max_new_tokens=max_new_tokens, instructions=instructions, system_prompt=system_prompt))

        os.makedirs(f"./output/{model_id}/{shots}", exist_ok = True)
        outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
        outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")

    def prompt_model_dataset(self, prompt_dataset, eval_dataset, model_id = "meta-llama/Llama-3.2-1B-Instruct", shots = 0, max_new_tokens = 1000, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
        """
        This function prompts the whole dataset to the model in a x-shot manner. Saves the generated text as a Huggingface dataset (not human readable) and as a json file (human readable).

        @params
            prompt_dataset: prompt samples
            eval_dataset: evaluation samples
            model_id: Huggingface model name
            shots: number of examples in the instructions
            max_new_tokens: limit of the number of generated tokens
            instructions: model instructions
            system_prompt: general system instructions 
        """
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        few_shots = self.create_few_shot_samples(eval_dataset, shots)
        
        #prompt_dataset = prompt_dataset.select([i for i in range(shots, len(prompt_dataset))]) # Remove few-shot-examples
        prompt_dataset = prompt_dataset.skip(shots)
        
        message_dataset = prompt_dataset.select(range(10)).map(lambda x: self.create_message_dataset(sample=x, few_shot=few_shots, instructions=instructions, system_prompt=system_prompt))

        outputs = pipe(message_dataset["messages"], max_new_tokens = max_new_tokens)

        print(outputs)
        # for out in tqdm(pipe(message_dataset)):
        #     print(out)
        os.makedirs(f"./output/{model_id}/{shots}", exist_ok=True)
        # outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
        with open(f"./output/{model_id}/{shots}/outputs_{model_id.split('/')[-1]}_{shots}.json", "w", encoding = "utf-8") as f:
            json.dump(outputs, f, indent=4)
        #outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")

    def sanity_check(self, answer, expected_keys):
        s = answer

        # Test if Dictionary is contained in the answer
        try:
            if s.find("{") != -1:

                s = s[s.find("{"):]

            if s.find("}") != -1:
                s = s[:s.find("}")+1]

            answer = str(s)
            #print("Found Dictionary:", s)

            # Save dict
            model_dict = ast.literal_eval(answer)
            #print("Parsed dict: ", model_dict)
            model_keys = set(model_dict.keys())
            if len(model_keys) == len(expected_keys)-1 and len(model_keys & expected_keys) == len(expected_keys)-1: # -1 to exclude Dokument_ID
                return model_dict
            else:
                print(f"Dictionary does not have the correct format:\n{answer}")
                print("expected keys:", expected_keys)
                print("model keys:", model_keys)
                print("missing keys:", (expected_keys-model_keys))
                return ""
        except:
            print(f"Answer not correct:\n{answer}")
            return ""
        
    def get_answer(self, chain, instructions, few_shots, prompt_sample, expected_keys):
        answer = chain.invoke({"instructions": instructions, "few_shots": few_shots, "prompt_sample": str(prompt_sample)})

        model_dict = self.sanity_check(answer, expected_keys)

        return answer, model_dict


    def prompt_llama_dataset(self, prompt_dataset, eval_dataset, system_prompt = DEFAULT_SYSTEM_PROMPT_DE, instructions = DEFAULT_PROMPT_DE, shots=0, experiment_mode = "dev", model_id = "llama3"):
        self.shots = shots
        self.model_path_id = ""
        if model_id.lower() == "llama3" or model_id.lower() == "llama3:8b":
            self.model_path_id = "Llama3_8B"
        elif model_id.lower() == "llama3:70b":
            self.model_path_id = "Llama3_70B"
        else:
            self.model_path_id = model_id

        self.prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
        ])

        few_shots = self.create_few_shot_samples(eval_dataset, shots)
        expected_keys = set(eval_dataset.features)
        #print("expected_keys:", expected_keys)
        
        prompt_dataset = prompt_dataset.skip(shots)

        if experiment_mode == "dev":
            prompt_dataset = prompt_dataset.select(range(10)) # only select development samples
        else:
            prompt_dataset = prompt_dataset.skip(10) # exclude development samples

        self.outputs = defaultdict(str)
        self.answers = defaultdict(list)
        self.answers["system_prompt"] = system_prompt
        self.answers["instructions"] = instructions

        self.ckp_file_outputs = f"./output/{self.model_path_id}/{experiment_mode}/{shots}/outputs_{self.model_path_id}_{experiment_mode}_{shots}_ckp.json"
        self.ckp_file_answers = f"./output/{self.model_path_id}/{experiment_mode}/{shots}/answers_{self.model_path_id}_{experiment_mode}_{shots}_ckp.json"
        #print("ckp_file:", ckp_file)
        self.ckp_outputs = self.load_ckp(self.ckp_file_outputs) # returns None or dict
        self.ckp_answers = self.load_ckp(self.ckp_file_answers)
        #print("ckp_data:", ckp_data)

        if self.ckp_outputs:
            prompts_w_ckps = prompt_dataset
            prompt_dataset = prompts_w_ckps.filter(lambda x: x["Dokument_ID"] not in self.ckp_outputs.keys())
            #print("first idx:", prompt_dataset["Dokument_ID"][0])
        else:
            self.ckp_outputs = defaultdict(str)
        
        if not self.ckp_answers:
            self.ckp_answers = defaultdict(list)
        
        # TODO: Über mapping lösen
        self.i = 0
        prompt_dataset.map(lambda x: self.prompt_sample(x, instructions, few_shots, expected_keys))
        # for i, prompt_sample in tqdm(enumerate(prompt_dataset)):
        #     if i == 0:
        #         print("Starting with sample: ", prompt_sample["Dokument_ID"])

        #     if "70" not in model_id:
        #         tries = 0
        #     else:
        #         tries = 2 # no retries, takes too long
        #     #print("curr sample: ", prompt_sample)
        #     template = "{instructions}{few_shots}Input: {prompt_sample}\nOutput: "

        #     prompt = ChatPromptTemplate.from_template(template)

        #     model = OllamaLLM(model=model_id, keep_alive = -1, num_threads = 32)

        #     chain = prompt | model

        #     answer, model_dict = self.get_answer(chain, instructions, few_shots, prompt_sample, expected_keys)
        #     answers[prompt_sample["Dokument_ID"]].append(answer)


        #     while model_dict == "" and tries < 2:
        #         print(f"Try {tries+1} failed, prompting again...")
        #         answer, model_dict = self.get_answer(chain, instructions, few_shots, prompt_sample, expected_keys)
        #         answers[prompt_sample["Dokument_ID"]].append(answer)
        #         tries += 1
            
        #     outputs[prompt_sample["Dokument_ID"]] = model_dict

        #     if ckp_outputs:
        #         ckp_outputs[prompt_sample["Dokument_ID"]] = model_dict
    
        #     if ckp_answers:
        #         ckp_answers[prompt_sample["Dokument_ID"]] = answers[prompt_sample["Dokument_ID"]]

        #     # Checkpoints
        #     if i%10 == 0:
        #         print(f"Saving checkpoint {i}...")
        #         os.makedirs(f"./output/{model_path_id}/{experiment_mode}/{shots}", exist_ok=True)
        #         with open(ckp_file_outputs, "w", encoding = "utf-8") as f:
        #             json.dump(ckp_outputs, f, indent=4)
        #         with open(ckp_file_answers, "w", encoding = "utf-8") as f:
        #             json.dump(ckp_answers, f, indent=4)


        #print(outputs)

        os.makedirs(f"./output/{self.model_path_id}/{experiment_mode}/{shots}", exist_ok=True)
        try:
            self.outputs.save_to_disk(f"./output/{self.model_path_id}/{experiment_mode}/{shots}/hf")
            self.outputs.to_json(f"./output/{self.model_path_id}/{experiment_mode}/{shots}/outputs_{self.model_path_id}_{experiment_mode}_{shots}.json")
        except:

            with open(f"./output/{self.model_path_id}/{experiment_mode}/{shots}/outputs_{self.model_path_id}_{experiment_mode}_{shots}.json", "w", encoding = "utf-8") as f:
                json.dump(self.outputs, f, indent=4)
            with open(f"./output/{self.model_path_id}/{experiment_mode}/{shots}/answers_{self.model_path_id}_{experiment_mode}_{shots}.json", "w", encoding = "utf-8") as f:
                json.dump(self.answers, f, indent=4)

    def load_ckp(self, ckp_file):
        try:
            with open(ckp_file, "r", encoding = "utf-8") as f:
                #print("loading")
                return json.load(f)
        except:
            return None
        
    def prompt_sample(self, prompt_sample, instructions, few_shots, expected_keys):
        if self.i == 0:
            print("Starting with sample: ", prompt_sample["Dokument_ID"])

        if "70" not in self.model_id:
            tries = 0
        else:
            tries = 2 # no retries, takes too long
        #print("curr sample: ", prompt_sample)
        template = "{instructions}{few_shots}Input: {prompt_sample}\nOutput: "

        self.prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model=self.model_id, keep_alive = -1, num_threads = 32)

        chain = self.prompt | model

        answer, model_dict = self.get_answer(chain, instructions, few_shots, prompt_sample, expected_keys)
        self.answers[prompt_sample["Dokument_ID"]].append(answer)

        while model_dict == "" and tries < 2:
            print(f"Try {tries+1} failed, prompting again...")
            answer, model_dict = self.get_answer(chain, instructions, few_shots, prompt_sample, expected_keys)
            self.answers[prompt_sample["Dokument_ID"]].append(answer)
            tries += 1
        
        self.outputs[prompt_sample["Dokument_ID"]] = model_dict

        self.ckp_outputs[prompt_sample["Dokument_ID"]] = model_dict
        self.ckp_answers[prompt_sample["Dokument_ID"]] = self.answers[prompt_sample["Dokument_ID"]]

        # Checkpoints
        if self.i%10 == 0:
            print(f"Saving checkpoint {self.i}...")
            os.makedirs(f"./output/{self.model_path_id}/{self.experiment_mode}/{self.shots}", exist_ok=True)
            with open(self.ckp_file_outputs, "w", encoding = "utf-8") as f:
                json.dump(self.ckp_outputs, f, indent=4)
            with open(self.ckp_file_answers, "w", encoding = "utf-8") as f:
                json.dump(self.ckp_answers, f, indent=4)
        
        self.i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', help='model to run the experiment with', default = "llama3")
    parser.add_argument('--experiment_mode', '-e', help='experiment mode', default = "dev")
    parser.add_argument('--shot', '-s', help='shot', default = None)
    model_name = parser.parse_args().model_name
    experiment_mode = parser.parse_args().experiment_mode
    arg_shot = parser.parse_args().shot

    data = Data()
    prompter = Prompter(model_id=model_name, experiment_mode=experiment_mode)

    if arg_shot:
        print(f"Processing {arg_shot}-shot prompts...")
        prompter.prompt_llama_dataset(prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots = int(arg_shot), model_id = model_name, experiment_mode=experiment_mode)
    else:
        shots = [0, 1, 5]
        for shot in shots:
            # llama3 = 8B
            print(f"Processing {shot}-shot prompts...")
            prompter.prompt_llama_dataset(prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots = shot, model_id = model_name, experiment_mode=experiment_mode)