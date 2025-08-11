import requests
import os
from data import Data

def scrape_API(sample, modus: str = "author"):
    """
    This method is a helpers-function to scrape the wiki-API.

    @params 
        sample: sample to collect information for
        modus: query for author or publication data
    @returns 
        dictionary with the collected information
    """
    S = requests.Session()

    URL = f"https://query.wikidata.org/bigdata/namespace/wdq/sparql"

    if not sample["Vorname"]:
        sample["Vorname"] = ""
    
    if not sample["Nachname"]:
        sample["Nachname"] = ""

    # define query
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
    elif modus == "author_sloppy":
        query = f'PREFIX prop: <http://www.wikidata.org/prop/direct/>\
        \
        SELECT DISTINCT ?info \
        {{\
            ?author ?p1 "{sample["Vorname"] + " " + sample["Nachname"]}" ;\
                    ?p2 ?info .\
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
    elif modus == "work_sloppy":
        query = f'PREFIX prop: <http://www.wikidata.org/prop/direct/>\
        \
        SELECT DISTINCT ?info \
        {{\
            ?work prop:P1476 "{sample["Titel"]}"@de ;\
                ?p1 ?info .\
        \
            FILTER(isLiteral(?info))\
        }}'

    PARAMS = {
    "query": query,
    "format": "json",
    }

    HEADERS = {'User-Agent': 'LLAMCoWikiBot/0.0 (j.grimm@campus.lmu.de) python-request/0.0'}

    # get query result
    R = S.get(url=URL, params=PARAMS, headers = HEADERS)
    
    try:
        DATA = R.json()

        values = filter_api_result(DATA)

        if len(values) == 0 and "sloppy" not in modus:
            return scrape_API(sample, modus = f"{modus}_sloppy")

        return {"Output": list(values)}
    except:
        print(f"Sample {sample["Dokument_ID"]} could not be processed:\n\t{R.reason}")

def filter_api_result(api_data):
    """
    This function returns the values of the query result.

    @params
        api_data: API result
    @returns
        values: list of the values in the query result
    """
    values = {entry["info"]["value"] for entry in api_data["results"]["bindings"]}
    return values

def fetch_info(modus: str = "author"):
    """
    This function queries the wikidata API.

    @params
        modus: query for author or publication data
    """
    print("Fetching corpus data...")
    corpus_data = Data()

    print(f"Posting SPARQL queries for '{modus}'...")
    outputs = corpus_data.prompt_samples.map(lambda x: scrape_API(sample = x, modus = modus))

    print("Saving output...")
    os.makedirs("./output/wikidata/hf", exist_ok=True)
    outputs.save_to_disk(f"./output/wikidata/hf/{modus}")
    outputs.to_json(f"./output/wikidata/{modus}_outputs_wikidata_hf.json")

if __name__ == "__main__":
    fetch_info(modus="author")
    fetch_info(modus="work")