import requests
import os
from data import Data
import json
import argparse
# https://query.wikidata.org/bigdata/namespace/wdq/sparql?query={SPARQL}

def scrape_API(sample, modus = "author"):
    """
    This method is a helpers-function to scrape the wiki-API.

    @params params: dictionary with parameters that should be used to scrape the API
    @returns DATA: a json-object with the results of the API-search
    """
    S = requests.Session()

    URL = f"https://query.wikidata.org/bigdata/namespace/wdq/sparql"

    if not sample["Vorname"]:
        sample["Vorname"] = ""
    
    if not sample["Nachname"]:
        sample["Nachname"] = ""

    query = ""
    if modus == "author":
        query = f'PREFIX prop: <http://www.wikidata.org/prop/direct/>\
        \
        SELECT DISTINCT ?info \
        {{\
            ?author ?p1 "{sample["Vorname"] + " " + sample["Nachname"]}" ;\
                    ?p2 ?work ;\
                    ?p3 ?info .\
        \
            ?work prop:P1476 "{sample["Titel"]}"@de .\
        \
            FILTER(isLiteral(?info))\
        }}'
    elif modus == "work":
        query = f'PREFIX prop: <http://www.wikidata.org/prop/direct/>\
        \
        SELECT DISTINCT ?info \ 
        {{\
            ?author ?p1 "{sample["Vorname"] + " " + sample["Nachname"]}" ;\
                ?p2 ?work .\
        \
            ?work prop:P1476 "{sample["Titel"]}"@de ;\
                ?p3 ?info .\
        \
            FILTER(isLiteral(?info))\
        }}'

    PARAMS = {
    "query": query,
    "format": "json",
    }

    HEADERS = {'User-Agent': 'LLAMCoWikiBot/0.0 (j.grimm@campus.lmu.de) python-request/0.0'}

    R = S.get(url=URL, params=PARAMS, headers = HEADERS)
    
    try:
        DATA = R.json()

        return filter_api_result(DATA)
    except:
        print(f"Sample {sample["Dokument_ID"]} could not be processed:\n\t{R.reason}")

def filter_api_result(api_data):
    values = {entry["info"]["value"] for entry in api_data["results"]["bindings"]}
    return {"output": list(values)}

def fetch_info(modus = "author"):
    corpus_data = Data()

    outputs = {sample["Dokument_ID"]: scrape_API(sample, modus=modus) for sample in corpus_data.prompt_samples}
    # outputs = corpus_data.prompt_samples.map(lambda x: scrape_API(sample = x))

    os.makedirs("./output/wikidata/", exist_ok=True)
    # outputs.save_to_disk("./output/wikidata/hf")
    #outputs.to_json("./output/wikidata/outputs_wikidata.json")

    # outputs.save_to_disc(f"./output/{model_id}/{shots}/hf")
    with open(f"./output/wikidata/{modus}_outputs_wikidata.json", "w", encoding = "utf-8") as f:
        json.dump(outputs, f)

if __name__ == "__main__":
    #fetch_info(modus="author")
    fetch_info(modus="work")

    # outputs = load_from_disk("./output/wikidata/hf")
    # print(outputs)