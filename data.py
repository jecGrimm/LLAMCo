import pandas as pd
from datasets import Dataset, load_from_disk
import re
import os

class Data:
    def __init__(self):
        # Create and specify dataset storage
        data_dir = "./data"
        prompt_dir = data_dir + "/prompt_samples/"
        eval_dir = data_dir + "/eval_samples/"
        os.makedirs(prompt_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        try:
           # try loading the datasets
           self.prompt_samples = load_from_disk(prompt_dir)
           self.eval_samples = load_from_disk(eval_dir)
        except: 
            # execute if the datasets are not yet stored locally
            self.whole_corpus = self.read_corpus() # whole metadata collection (including episode entries)
            self.whole_idx_corpus = self.whole_corpus.filter(lambda x: re.match(r"^0+$", x["Dokument_ID"].split('-')[-1])) # filter out episode entries

            # create and store prompt dataset
            self.prompt_samples = self.create_prompt_samples()
            self.prompt_samples.save_to_disk(prompt_dir)

            # create and store evaluation dataset
            self.eval_samples = self.create_evaluation_samples()
            self.eval_samples.save_to_disk(eval_dir)

    def read_corpus(self):
        """
        This method reads the metadata from the file Bibliographie.tsv.

        @returns Huggingface dataset containing the whole corpus 
        """
        os.makedirs("resources", exist_ok=True)
        df = pd.read_csv("resources/Bibliographie.tsv", sep = '\t', header = 0)
        return Dataset.from_pandas(df)
    
    def create_prompt_samples(self):
        """
        This method extracts the columns from the corpus that are needed for the prompts.

        @returns Huggingface dataset containing only the columns needed for the prompt samples
        """
        rem_feats = set(self.whole_idx_corpus.features) - set(["Dokument_ID", "Vorname", "Nachname", "Titel"])
        return self.whole_idx_corpus.remove_columns(rem_feats)
    
    def create_evaluation_samples(self):
        """
        This method extracts the columns from the corpus that are needed for the evaluation.

        @returns Huggingface dataset containing only the columns needed for the evaluation samples
        """
        rem_feats = [\
            "Gattungslabel_ED_normalisiert",\
            "Nummer im Heft (ab 00797: 1 erste Position. 0 nicht erste Position)",\
            "Verantwortlich_Erfassung",\
            "falls andere Quelle",\
            "Herkunft_Prätext_lt_Paratext",\
            "Wahrheitsbeglaubigung_lt_Paratext",\
            "Bearbeitungsqualität",\
            "ist_spätere_Fassung_von",\
            "UrhGeschBis",\
            "falls_Episode_als_Ganztext_erfasst",\
            "Kontrafaktizität_annotiert"\
        ]
        return self.whole_idx_corpus.remove_columns(rem_feats)
    
    
if __name__ == "__main__":
    data = Data()
    print("Done")