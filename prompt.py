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

# TODO: Unsichere Prompts ausprobieren

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
Bitte stelle sicher, dass deine Antworten keine sozialen Vorurteile widerspiegeln und bleibe positiv.\n\
\n\
Wenn eine Frage keinen Sinn macht oder faktisch nicht kohärent ist, erkläre bitte, warum, statt eine falsche Antwort zu geben.\n\
Wenn du die Antwort auf eine Frage nicht kennst, teile bitte keine falschen Informationen und antworte mit "NAN".\n\
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

DEFAULT_PROMPT_DE_0 = """\
Du bist ein wissenschaftlicher Assistent, der Metadaten über die Veröffentlichung eines literarischen Textes sammelt.\n\
Erstelle ein Python-Dictionary, das nur die gegebenen Output Keys enthält und diese den entsprechenden Informationen des Autors und Titels aus dem Input zuweist.\n\
Bitte sammle die Informationen selbst entweder aus deinem Wissen oder mit einer Websuche.\n\
Bitte nutze die Erklärungen für die Output Keys, um die korrekten Informationen zu erfassen.\n\
Bitte fülle als Wert "" ein, wenn du eine Information nicht finden kannst.\n\
Bitte gib nur das Output Dictionary zurück. Bitte gib das Dictionary wie im Output Format angegeben zurück.\n\
Bitte antworte auf Deutsch.\n\
\n\
Input Keys:\n\
"Vorname": Vorname des Autors\n\
"Nachname": Nachname des Autors\n\
"Titel": Titel des Werks, das von dem Autor geschrieben wurde\n\
\n\
Output Keys:\n\
"Vorname": Vorname des Autors aus dem Input\n\
"Nachname": Nachname des Autors aus dem Input\n\
"Pseudonym": Pseudonym des Autors aus dem Input\n\
"Gender": Angenommenes Gender des Autors aus dem Input; mögliche Werte sind "m" (männlich), "f" (weiblich)\n\
"Titel": Titel aus dem Input\n\
"Untertitel_im_Text": Untertitel des gedruckten Textes aus dem Input\n\
"Untertitel_im_Inhaltsverzeichnis": Untertitel des Textes aus dem Input im Inhaltsverzeichnis der Veröffentlichung\n\
"Jahr_ED": Jahr der ersten Veröffentlichung des Textes aus dem Input\n\
"entstanden": Jahr der Entstehung des Textes aus dem Input\n\
"Gattungslabel_ED":	Genrebezeichnung bei der Erstveröffentlichung des Textes aus dem Input\n\
"Medium_ED": Medium der Erstveröffentlichung des Textes aus dem Input\n\
"Medientyp_ED": Typ des Mediums der Erstveröffentlichung des Textes aus dem Input\n\
"Hg.": Herausgeber der Erstveröffentlichung des Textes aus dem Input\n\
"Kanon_Status":	Kanonstatus des Autors aus dem Input; mögliche Werte sind 0 (vergessener Autor, Autor ist nicht Teil des literarischen Kanons, keine digitalisierten Texte des Autors sind im Internet verfügbar), 1 (heute vergessener Autor, Autor ist nicht Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar), 2 (bekannter Autor, Autor ist nicht Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar), 3 (bekannter Autor, Autor ist Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar)\n\
"seriell": ob der Text aus dem Input seriell veröffentlicht wurde, mögliche Werte sind True (der Text aus dem Input wurde seriell veröffentlicht) und False (der Text aus dem Input wurde nicht seriell veröffentlicht)\n\
"Seiten": Seitenzahlen der Erstveröffentlichung des Textes aus dem Input; das Format der Seitenzahlen ist <erste Seite>-<letzte Seite>\n\
"Medium_Zweitdruck": Medium der Zweitveröffentlichung des Textes aus dem Input\n\
"Jahr_Zweitdruck": Jahr der zweiten Veröffentlichung des Textes aus dem Input\n\
"Label_Zweitdruck": Genre der Zweitveröffentlichung des Textes aus dem Input\n\
"Medium_Drittdruck": Medium der Drittveröffentlichung des Textes aus dem Input\n\
"Jahr_Drittdruck": Jahr der dritten Veröffentlichung des Textes aus dem Input\n\
"Label_Drittdruck": Genre der Drittveröffentlichung des Textes aus dem Input\n\
"in_Deutscher_Novellenschatz_(Heyse)": ob der Text aus dem Input in "Deutscher Novellenschatz" von Paul Heyse aufgelistet wird; mögliche Werte sind True (der Text aus dem Input wird in "Deutscher Novellenschatz" von Paul Heyse gelistet) und False (der Text aus dem Input wird nicht in "Deutscher Novellenschatz" von Paul Heyse gelistet)\n\
"in_Pantheon": ob der Text aus dem Input in "Pantheon" von Carl Hoffmann (Herausgeber) aufgelistet wird; mögliche Werte sind True (der Text aus dem Input wird in "Pantheon" von Carl Hoffmann (Herausgeber) gelistet) und False (der Text aus dem Input wird nicht in "Pantheon" von Carl Hoffmann (Herausgeber) gelistet)\n\
"in_B-v-Wiese": ob der Text aus dem Input in "Novelle" von Benno von Wiese aufgelistet wird; mögliche Werte sind True (der Text aus dem Input wird in "Novelle" von Benno von Wiese gelistet) und False (der Text aus dem Input wird nicht in "Novelle" von Benno von Wiese gelistet)\n\
\n\
Output Format:\n\
{\n\
    "Vorname": "",\n\
    "Nachname": "",\n\
    "Pseudonym": "",\n\
    "Gender": "",\n\
    "Titel": "",\n\
    "Untertitel_im_Text": "",\n\
    "Untertitel_im_Inhaltsverzeichnis": "",\n\
    "Jahr_ED": "",\n\
    "entstanden": "",\n\
    "Gattungslabel_ED": "",\n\
    "Medium_ED": "",\n\
    "Medientyp_ED": "",\n\
    "Hg.": "",\n\
    "Kanon_Status": "",\n\
    "seriell": "",\n\
    "Seiten": "",\n\
    "Medium_Zweitdruck": "",\n\
    "Jahr_Zweitdruck": "",\n\
    "Label_Zweitdruck": "",\n\
    "Medium_Drittdruck": "",\n\
    "Jahr_Drittdruck": "",\n\
    "Label_Drittdruck": "",\n\
    "in_Deutscher_Novellenschatz_(Heyse)": "",\n\
    "in_Pantheon": "",\n\
    "in_B-v-Wiese": "",\n\
}\n\
\n\
"""

DEFAULT_PROMPT_DE_1 = """\
Du bist ein wissenschaftlicher Assistent, der Metadaten über die Veröffentlichung eines literarischen Textes sammelt.\n\
Erstelle ein Python-Dictionary, das nur die gegebenen Output Keys enthält und diese den entsprechenden Informationen des Autors und Titels aus dem Input zuweist.\n\
Bitte sammle die Informationen selbst entweder aus deinem Wissen oder mit einer Websuche.\n\
Bitte nutze die Erklärungen für die Output Keys, um die korrekten Informationen zu erfassen.\n\
Bitte fülle als Wert "" ein, wenn du eine Information nicht finden kannst.\n\
Bitte gib nur das Output Dictionary zurück. Bitte gib das Dictionary wie im Output Format angegeben zurück.\n\
Bitte antworte auf Deutsch.\n\
\n\
Input Keys:\n\
"Vorname": Vorname des Autors\n\
"Nachname": Nachname des Autors\n\
"Titel": Titel des Werks, das von dem Autor geschrieben wurde\n\
\n\
Output Keys:\n\
"Vorname": Vorname des Autors\n\
"Nachname": Nachname des Autors\n\
"Pseudonym": Pseudonym des Autors\n\
"Gender": Angenommenes Gender des Autors; mögliche Werte sind "m" (männlich), "f" (weiblich)\n\
"Titel": Titel\n\
"Untertitel_im_Text": Untertitel des gedruckten Textes\n\
"Untertitel_im_Inhaltsverzeichnis": Untertitel des Textes im Inhaltsverzeichnis der Veröffentlichung\n\
"Jahr_ED": Jahr der ersten Veröffentlichung des Textes\n\
"entstanden": Jahr der Entstehung des Textes\n\
"Gattungslabel_ED":	Genrebezeichnung bei der Erstveröffentlichung des Textes\n\
"Medium_ED": Medium der Erstveröffentlichung des Textes\n\
"Medientyp_ED": Typ des Mediums der Erstveröffentlichung des Textes\n\
"Hg.": Herausgeber der Erstveröffentlichung des Textes\n\
"Kanon_Status":	Kanonstatus des Autors; mögliche Werte sind 0 (vergessener Autor, Autor ist nicht Teil des literarischen Kanons, keine digitalisierten Texte des Autors sind im Internet verfügbar), 1 (heute vergessener Autor, Autor ist nicht Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar), 2 (bekannter Autor, Autor ist nicht Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar), 3 (bekannter Autor, Autor ist Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar)\n\
"seriell": ob der Text seriell veröffentlicht wurde, mögliche Werte sind True (der Text wurde seriell veröffentlicht) und False (der Text wurde nicht seriell veröffentlicht)\n\
"Seiten": Seitenzahlen der Erstveröffentlichung des Textes; das Format der Seitenzahlen ist <erste Seite>-<letzte Seite>\n\
"Medium_Zweitdruck": Medium der Zweitveröffentlichung des Textes\n\
"Jahr_Zweitdruck": Jahr der zweiten Veröffentlichung des Textes\n\
"Label_Zweitdruck": Genre der Zweitveröffentlichung des Textes\n\
"Medium_Drittdruck": Medium der Drittveröffentlichung des Textes\n\
"Jahr_Drittdruck": Jahr der dritten Veröffentlichung des Textes\n\
"Label_Drittdruck": Genre der Drittveröffentlichung des Textes\n\
"in_Deutscher_Novellenschatz_(Heyse)": ob der Text in "Deutscher Novellenschatz" von Paul Heyse aufgelistet wird; mögliche Werte sind True (der Text wird in "Deutscher Novellenschatz" von Paul Heyse gelistet) und False (der Text wird nicht in "Deutscher Novellenschatz" von Paul Heyse gelistet)\n\
"in_Pantheon": ob der Text in "Pantheon" von Carl Hoffmann (Herausgeber) aufgelistet wird; mögliche Werte sind True (der Text wird in "Pantheon" von Carl Hoffmann (Herausgeber) gelistet) und False (der Text wird nicht in "Pantheon" von Carl Hoffmann (Herausgeber) gelistet)\n\
"in_B-v-Wiese": ob der Text in "Novelle" von Benno von Wiese aufgelistet wird; mögliche Werte sind True (der Text wird in "Novelle" von Benno von Wiese gelistet) und False (der Text wird nicht in "Novelle" von Benno von Wiese gelistet)\n\
\n\
Output Format:\n\
{\n\
    "Vorname": "",\n\
    "Nachname": "",\n\
    "Pseudonym": "",\n\
    "Gender": "",\n\
    "Titel": "",\n\
    "Untertitel_im_Text": "",\n\
    "Untertitel_im_Inhaltsverzeichnis": "",\n\
    "Jahr_ED": "",\n\
    "entstanden": "",\n\
    "Gattungslabel_ED": "",\n\
    "Medium_ED": "",\n\
    "Medientyp_ED": "",\n\
    "Hg.": "",\n\
    "Kanon_Status": "",\n\
    "seriell": "",\n\
    "Seiten": "",\n\
    "Medium_Zweitdruck": "",\n\
    "Jahr_Zweitdruck": "",\n\
    "Label_Zweitdruck": "",\n\
    "Medium_Drittdruck": "",\n\
    "Jahr_Drittdruck": "",\n\
    "Label_Drittdruck": "",\n\
    "in_Deutscher_Novellenschatz_(Heyse)": "",\n\
    "in_Pantheon": "",\n\
    "in_B-v-Wiese": "",\n\
}\n\
\n\
"""

DEFAULT_PROMPT_DE = """\
Du bist ein wissenschaftlicher Assistent, der Metadaten über die Veröffentlichung eines literarischen Textes sammelt.\n\
Bitte sammle die Informationen selbst entweder aus deinem Wissen oder mit einer Websuche.\n\
Bitte nutze die Erklärungen für die Output Keys, um die korrekten Informationen zu erfassen.\n\
Bitte fülle als Wert "" ein, wenn du eine Information nicht finden kannst.\n\
Bitte gib nur das ein Dictionary zurück.\n\
Bitte antworte auf Deutsch.\n\
\n\
Input Keys:\n\
"Vorname": Vorname des Autors\n\
"Nachname": Nachname des Autors\n\
"Titel": Titel des Werks, das von dem Autor geschrieben wurde\n\
\n\
Output Keys:\n\
"Vorname": Vorname des Autors\n\
"Nachname": Nachname des Autors\n\
"Pseudonym": Pseudonym des Autors\n\
"Gender": Angenommenes Gender des Autors; mögliche Werte sind "m" (männlich), "f" (weiblich)\n\
"Titel": Titel\n\
"Untertitel_im_Text": Untertitel des gedruckten Textes\n\
"Untertitel_im_Inhaltsverzeichnis": Untertitel des Textes im Inhaltsverzeichnis der Veröffentlichung\n\
"Jahr_ED": Jahr der ersten Veröffentlichung des Textes\n\
"entstanden": Jahr der Entstehung des Textes\n\
"Gattungslabel_ED":	Genrebezeichnung bei der Erstveröffentlichung des Textes\n\
"Medium_ED": Medium der Erstveröffentlichung des Textes\n\
"Medientyp_ED": Typ des Mediums der Erstveröffentlichung des Textes\n\
"Hg.": Herausgeber der Erstveröffentlichung des Textes\n\
"Kanon_Status":	Kanonstatus des Autors; mögliche Werte sind 0 (vergessener Autor, Autor ist nicht Teil des literarischen Kanons, keine digitalisierten Texte des Autors sind im Internet verfügbar), 1 (heute vergessener Autor, Autor ist nicht Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar), 2 (bekannter Autor, Autor ist nicht Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar), 3 (bekannter Autor, Autor ist Teil des literarischen Kanons, digitalisierte Texte des Autors sind im Internet verfügbar)\n\
"seriell": ob der Text seriell veröffentlicht wurde, mögliche Werte sind True (der Text wurde seriell veröffentlicht) und False (der Text wurde nicht seriell veröffentlicht)\n\
"Seiten": Seitenzahlen der Erstveröffentlichung des Textes; das Format der Seitenzahlen ist <erste Seite>-<letzte Seite>\n\
"Medium_Zweitdruck": Medium der Zweitveröffentlichung des Textes\n\
"Jahr_Zweitdruck": Jahr der zweiten Veröffentlichung des Textes\n\
"Label_Zweitdruck": Genre der Zweitveröffentlichung des Textes\n\
"Medium_Drittdruck": Medium der Drittveröffentlichung des Textes\n\
"Jahr_Drittdruck": Jahr der dritten Veröffentlichung des Textes\n\
"Label_Drittdruck": Genre der Drittveröffentlichung des Textes\n\
"in_Deutscher_Novellenschatz_(Heyse)": ob der Text in "Deutscher Novellenschatz" von Paul Heyse aufgelistet wird; mögliche Werte sind True (der Text wird in "Deutscher Novellenschatz" von Paul Heyse gelistet) und False (der Text wird nicht in "Deutscher Novellenschatz" von Paul Heyse gelistet)\n\
"in_Pantheon": ob der Text in "Pantheon" von Carl Hoffmann (Herausgeber) aufgelistet wird; mögliche Werte sind True (der Text wird in "Pantheon" von Carl Hoffmann (Herausgeber) gelistet) und False (der Text wird nicht in "Pantheon" von Carl Hoffmann (Herausgeber) gelistet)\n\
"in_B-v-Wiese": ob der Text in "Novelle" von Benno von Wiese aufgelistet wird; mögliche Werte sind True (der Text wird in "Novelle" von Benno von Wiese gelistet) und False (der Text wird nicht in "Novelle" von Benno von Wiese gelistet)\n\
\n\
"""

def create_messages(prompt, system_prompt = DEFAULT_SYSTEM_PROMPT_DE):
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

def create_prompt(prompt_sample, few_shots, instructions=DEFAULT_PROMPT_DE):
    """
    This function combines the parts of the prompt.

    @params
        prompt_sample: current sample 
        few_shots: x-shot samples
        instructions: prompt instructions
    @returns full prompt consisting of the instruction, x examples and the current sample
    """
    return instructions + few_shots + f"Input: {prompt_sample}\nOutput: "

def generate_text(sample, pipe, few_shot, max_new_tokens = 500, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
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

def create_message_dataset(sample, few_shot, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
    prompt = create_prompt(sample, few_shots=few_shot, instructions=instructions)
    messages = create_messages(prompt, system_prompt=system_prompt)
    return {"messages": messages}

def prompt_model(prompt_dataset, eval_dataset, model_id = "meta-llama/Llama-3.2-1B-Instruct", shots = 0, max_new_tokens = 500, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
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

    few_shots = create_few_shot_samples(eval_dataset, shots)
    
    prompt_dataset = prompt_dataset.select([i for i in range(shots, len(prompt_dataset))]) # Remove few-shot-examples
    
    outputs = prompt_dataset.map(lambda x: generate_text(sample=x, pipe=pipe, few_shot=few_shots, max_new_tokens=max_new_tokens, instructions=instructions, system_prompt=system_prompt))

    os.makedirs(f"./output/{model_id}/{shots}", exist_ok = True)
    outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
    outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")

def prompt_model_dataset(prompt_dataset, eval_dataset, model_id = "meta-llama/Llama-3.2-1B-Instruct", shots = 0, max_new_tokens = 1000, instructions=DEFAULT_PROMPT_DE, system_prompt=DEFAULT_SYSTEM_PROMPT_DE):
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

    outputs = pipe(message_dataset["messages"], max_new_tokens = max_new_tokens)

    print(outputs)
    # for out in tqdm(pipe(message_dataset)):
    #     print(out)
    os.makedirs(f"./output/{model_id}/{shots}", exist_ok=True)
    # outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
    with open(f"./output/{model_id}/{shots}/outputs_{model_id.split('/')[-1]}_{shots}.json", "w", encoding = "utf-8") as f:
        json.dump(outputs, f)
    #outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")

def sanity_check(answer, expected_keys):
    s = answer

    # Test if Dictionary is contained in the answer
    # TODO: add retry
    try:
        if s.find("{") != -1:

            s = s[s.find("{"):]

        if s.find("}") != -1:
            s = s[:s.find("}")+1]

        answer = s

        # Save dict
        model_dict = ast.literal_eval(answer)
        model_keys = set(model_dict.keys())
        if len(model_keys) == len(expected_keys)-1 and len(model_keys & expected_keys) == len(expected_keys)-1: # -1 to exclude Dokument_ID
            return model_dict
        else:
            print(f"Dictionary does not have the correct format:\n{answer}")
            print("model keys:", model_keys)
            return ""
    except:
        print(f"Answer not correct:\n{answer}")
        return ""
        #answer = answer

    #print(answer)

    # Store Dict
    # model_dict = ast.literal_eval(answer)
    # model_keys = set(model_dict.keys())
    # if len(model_keys) == len(expected_keys) and len(model_keys & expected_keys) == len(expected_keys):
    #     return model_dict
    # else:
    #     return ""
    
def get_answer(chain, instructions, few_shots, prompt_sample, expected_keys):
    answer = chain.invoke({"instructions": instructions, "few_shots": few_shots, "prompt_sample": str(prompt_sample)})

    model_dict = sanity_check(answer, expected_keys)

    return answer, model_dict


def prompt_llama8b_dataset(prompt_dataset, eval_dataset, system_prompt = DEFAULT_SYSTEM_PROMPT_DE, instructions = DEFAULT_PROMPT_DE, shots=0, experiment_mode = "dev"):
    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
    ])

    few_shots = create_few_shot_samples(eval_dataset, shots)
    expected_keys = set(eval_dataset.features)
    #print("expected_keys:", expected_keys)
    
    prompt_dataset = prompt_dataset.skip(shots)

    if experiment_mode == "dev":
        prompt_dataset = prompt_dataset.select(range(10)) # only select development samples
    else:
        prompt_dataset = prompt_dataset.skip(10) # exclude development samples

    outputs = defaultdict(str)
    answers = defaultdict(list)
    answers["system_prompt"] = system_prompt
    answers["instructions"] = instructions
	
    # TODO: Über mapping lösen
    for i, prompt_sample in enumerate(prompt_dataset):
        tries = 0
        #print("curr sample: ", prompt_sample)
        template = "{instructions}{few_shots}Input: {prompt_sample}\nOutput: "

        prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model="llama3")

        chain = prompt | model

        answer, model_dict = get_answer(chain, instructions, few_shots, prompt_sample, expected_keys)
        answers[prompt_sample["Dokument_ID"]].append(answer)

        while model_dict == "" and tries < 2:
            print(f"Try {tries+1} failed, prompting again...")
            answer, model_dict = get_answer(chain, instructions, few_shots, prompt_sample, expected_keys)
            answers[prompt_sample["Dokument_ID"]].append(answer)
            tries += 1
        
        outputs[prompt_sample["Dokument_ID"]] = model_dict

    #print(outputs)

    model_id = "Llama_8B"
    os.makedirs(f"./output/{model_id}/{shots}", exist_ok=True)
    try:
        outputs.save_to_disk(f"./output/{model_id}/{shots}/hf")
        outputs.to_json(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json")
    except:

        with open(f"./output/{model_id}/{shots}/outputs_{model_id}_{shots}.json", "w", encoding = "utf-8") as f:
            json.dump(outputs, f)
        with open(f"./output/{model_id}/{shots}/answers_{model_id}_{shots}.json", "w", encoding = "utf-8") as f:
            json.dump(answers, f)

    

if __name__ == "__main__":
    data = Data()
    shots = [0, 1, 5]
    for shot in shots:
        #prompt_model(prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots=shot)
        #prompt_model_dataset(model_id = "meta-llama/Llama-3.1-8B-Instruct", prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots=shot)
        prompt_llama8b_dataset(prompt_dataset=data.prompt_samples, eval_dataset=data.eval_samples, shots = shot)
