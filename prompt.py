import torch
from transformers import pipeline
import os
from data import Data
from tqdm.auto import tqdm
import json

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\
If you don’t know the answer to a question, please don’t share false information.\
If you don’t know the answer to a question, please answer with "NAN".\
"""

DEFAULT_PROMPT = """\
Please help me to collect structured metadata information about a published literary text.
Create a dictionary containing only the given output keys and the corresponding missing information for the author and title given by the input keys.\
Please refer to the explanations of the output keys to fill in the correct information.\
Please return "" as value if you don't know the answer.\
Please return only the output dictionary.\
Please answer in German.\
\
Input keys:\
"author": name of an author\
"title": name of a creative work written by the author\
\
Output keys:\
"last_name": last name of the author given in the input\
"first_name": first names of the author given in the input\
"pseudonym": pseudonym of the author given in the input\
"gender": assumed gender of the author given in the input; possible values "m" (male), "f" (female)\
"title": title of the creative work given in the input\
"subtitle in the text": subtitle in the printed text of the creative work given in the input\
"subtitle in the table of contents": subtitle in the table of contents of the creative work given in the input\
"year_first_publication": year of the first publication of the creative work given in the input\
"developed": year of the development of the creative work given in the input\
"genre_first_publication": genre of the first publication of the creative work given in the input\
"medium_first_publication": medium of the first publication of the creative work given in the input\
"medium_type_first_publication": type of the medium of the first publication of the creative work given in the input\
"publisher": publisher of the first publication of the creative work given in the input\
"canon_status": canon status of the author given in the input; possible values are 0 (unknown author, author not part of the literary canon, no digitalized texts available on the internet), 1 (author unknown today, author not part of the literary canon, digitalized texts available on the internet), 2 (author known, author not part of the literary canon, digitalized texts available on the internet), 3 (author known, author part of the literary canon, digitalized texts available on the internet)\
"serial": if the creative work given in the input was published serially; possible values are True (creative work given in the input was published serially), False (creative work given in the input was not published serially)\
"pages": page numbers of the first publication of the creative work given in the input; format of the values is <first_page_number>-<last_page_number>\
"medium_second_publication": medium of the second publication of the creative work given in the input\
"year_second_publication": year of the second publication of the creative work given in the input\
"genre_second_publication": genre of the second publication of the creative work given in the input\
"medium_third_publication": medium of the third publication of the creative work given in the input\
"year_third_publication": year of the third publication of the creative work given in the input\
"genre_third_publication": genre of the third publication of the creative work given in the input\
"in_Deutscher_Novellenschatz_Heyse": if the creative work given in the input is listed in "Deutscher Novellenschatz" by Paul Heyse; possible values are True (creative work given in the input is listed in "Deutscher Novellenschatz" by Paul Heyse), False (creative work given in the input is not listed in "Deutscher Novellenschatz" by Paul Heyse)\
"in_Pantheon_Hoffmann": if the creative work given in the input is listed in "Pantheon" by Carl Hoffmann (publisher); possible values are True (creative work given in the input is listed in "Pantheon" by Carl Hoffmann (publisher)), False (creative work given in the input is not listed in "Pantheon" by Carl Hoffmann (publisher))\
"in_Novelle_von_Wiese": if the creative work given in the input is listed in "Novelle" by Benno von Wiese; possible values are True (creative work given in the input is listed in "Novelle" by Benno von Wiese), False (creative work given in the input is not listed in "Novelle" by Benno von Wiese)\
\
"""

def create_messages(prompt, system_prompt = DEFAULT_SYSTEM_PROMPT):
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

def create_few_shot_samples(eval_dataset, shots = 0):
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

def create_prompt(prompt_sample, few_shots, instructions=DEFAULT_PROMPT):
    """
    This function combines the parts of the prompt.

    @params
        prompt_sample: current sample 
        few_shots: x-shot samples
        instructions: prompt instructions
    @returns full prompt consisting of the instruction, x examples and the current sample
    """
    return instructions + few_shots + f"Input: {prompt_sample}\nOutput: "

def generate_text(sample, pipe, few_shot, max_new_tokens = 500, instructions=DEFAULT_PROMPT, system_prompt=DEFAULT_SYSTEM_PROMPT):
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
    prompt = create_prompt(sample, few_shots=few_shot, instructions=instructions)
    messages = create_messages(prompt, system_prompt=system_prompt)

    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
    )
    # TODO: check for correctness + 3 tries to get it correct
    return {"outputs": outputs[0]["generated_text"][-1]} # dictionary to turn it automatically into a Huggingface dataset

def create_message_dataset(sample, few_shot, instructions=DEFAULT_PROMPT, system_prompt=DEFAULT_SYSTEM_PROMPT):
    prompt = create_prompt(sample, few_shots=few_shot, instructions=instructions)
    messages = create_messages(prompt, system_prompt=system_prompt)
    return {"messages": messages}

def prompt_model(prompt_dataset, eval_dataset, model_id = "meta-llama/Llama-3.2-1B-Instruct", shots = 0, max_new_tokens = 500, instructions=DEFAULT_PROMPT, system_prompt=DEFAULT_SYSTEM_PROMPT):
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

    few_shots = create_few_shot_samples(eval_dataset, shots)
    
    prompt_dataset = prompt_dataset.select([i for i in range(shots, len(prompt_dataset))]) # Remove few-shot-examples
    
    outputs = prompt_dataset.map(lambda x: generate_text(sample=x, pipe=pipe, few_shot=few_shots, max_new_tokens=max_new_tokens, instructions=instructions, system_prompt=system_prompt))

    os.makedirs(f"./output/{model_id}/{shots}", exist_ok = True)
    outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
    outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")

def prompt_model_dataset(prompt_dataset, eval_dataset, model_id = "meta-llama/Llama-3.2-1B-Instruct", shots = 0, max_new_tokens = 500, instructions=DEFAULT_PROMPT, system_prompt=DEFAULT_SYSTEM_PROMPT):
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

    few_shots = create_few_shot_samples(eval_dataset, shots)
    
    #prompt_dataset = prompt_dataset.select([i for i in range(shots, len(prompt_dataset))]) # Remove few-shot-examples
    prompt_dataset = prompt_dataset.skip(shots)
	
    message_dataset = prompt_dataset.select(range(10)).map(lambda x: create_message_dataset(sample=x, few_shot=few_shots, instructions=instructions, system_prompt=system_prompt))

    outputs = pipe(message_dataset["messages"])

    print(outputs)
    # for out in tqdm(pipe(message_dataset)):
    #     print(out)
    os.makedirs(f"./output/{model_id}/{shots}", exist_ok=True)
    # outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
    with open(f"./output/{model_id}/{shots}/outputs_{model_id.split('/')[-1]}_{shots}.json", "w", encoding = "utf-8") as f:
        json.dump(outputs, f)
    #outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")


if __name__ == "__main__":
    data = Data()
    shots = [0, 1, 5]
    for shot in shots:
	#print("Data:", data.prompt_samples[:10])
        #prompt_model(prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots=shot)
        prompt_model_dataset(prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots=shot)
