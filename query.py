import requests
import os
from data import Data
from datasets import load_from_disk
# https://query.wikidata.org/bigdata/namespace/wdq/sparql?query={SPARQL}

def scrape_API(sample):
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

    author_query = f'PREFIX prop: <http://www.wikidata.org/prop/direct/>\
    \
    Select DISTINCT ?author_info \
    {{\
        ?author ?p1 "{sample["Vorname"] + " " + sample["Nachname"]}" ;\
                ?p2 ?work ;\
                ?p3 ?author_info .\
    \
        ?work prop:P1476 "{sample["Titel"]}"@de .\
    \
        FILTER(isLiteral(?author_info))\
    }}'

    PARAMS = {
    "query": author_query,
    "format": "json",
    }

    R = S.get(url=URL, params=PARAMS)
    
    try:
        DATA = R.json()

        return filter_api_result(DATA)
    except:
        print(f"Sample {sample["Dokument_ID"]} could not be processed:\n\t{R.reason}")

def filter_api_result(api_data):
    values = {entry["author_info"]["value"] for entry in api_data["results"]["bindings"]}
    return {"output": list(values)}

def fetch_info():
    corpus_data = Data()

    outputs = {sample["Dokument_ID"]: scrape_API(sample) for sample in corpus_data.prompt_samples}
    # outputs = corpus_data.prompt_samples.map(lambda x: scrape_API(sample = x))

    # os.makedirs("./output/wikidata/", exist_ok=True)
    # outputs.save_to_disk("./output/wikidata/hf")
    outputs.to_json("./output/wikidata/outputs_wikidata.json")

if __name__ == "__main__":
    fetch_info()

    # outputs = load_from_disk("./output/wikidata/hf")
    # print(outputs)