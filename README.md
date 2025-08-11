# LLAMCo (LLMs As Metadata Collectors)
This repository examines whether Large Language Models (LLMs) can be used as metadata collectors. We use an exisiting metadata corpus for German 19th century literature and test whether LLMs are able to collect the same metadata information via prompting strategies.

## Installation
1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/index.html).
2. Create environment:<br> `conda env create -f env.yml`
3. Activate environment:<br> `conda activate llamco`

## Structure
### Additional Experiments
This directory contains the output of Llama-3.2-1B-Instruct.

### data
This directory contains the Huggingface datasets used for the experiments.
#### eval_samples
This directory contains the Huggingface dataset with the evaluation samples.
#### prompt_samples
This directory contains the Huggingface dataset with the prompt samples.

### notebooks
This directory contains notebooks that are not part of the experiments.
#### explore_Llama.ipynb
This file is a notebook to explore the experimental setup on Llama-3.2-1B-Instruct for singular examples.

### output
This directory contains the outputs and evaluation scors for each model (ChatGPT, Llama3-8B, Llama3-70B, wikidata). Each model directory entails the outputs for the dev and test split per x-shot experiment. The development outputs contain the results for each tested prompt.
#### evaluation_{model_id}\_{data_split}_{x_shot}_cols.json
These files contain the metrics calculated per column.
#### evaluation_{model_id}\_{data_split}_{x_shot}_rows.json
These files contain the metrics calculated per row.
#### evaluation_{model_id}\_{data_split}_{x_shot}_total.json
These files contain the metrics calculated for all cells in the corpus.
#### answers_{model_id}\_{data_split}_{x_shot}.json
These files contain all full answers by the models. 
#### outputs_{model_id}\_{data_split}_{x_shot}.json
These files contain the extracted dictionaries from the answers.
#### evaluation_{model_id}\_{data_split}_cols.png
These files visualize the recall results per column for one model and all x-shot settings.
#### wikidata
The wikidata outputs are not divided by data split or x-shot setting but by query as they either collect information about the author or the publication. 

### queries
This directory contains the SPARQL-queries used to collect the information from wikidata.

### resources
This directory contains resources deployed for the experiments.
#### Bibliographie.tsv
Metadata corpus used as the evaluation corpus.
#### prompts.txt
List of the manually engineered prompts tested in the development phase.  

### scripts
This directory contains the scripts and logs for the usage with SLURM. 

### data.py
Script that reads the data from the metadata corpus, transforms it into prompt and evaluation samples. It also stores or loads the already saved samples as a Huggingface dataset from or in the folder `data`.

### env.yml
Environment file for a quick installation of the needed dependencies via conda. See Section `Installation` for a guide how to create a new environment with this file.

### evaluate.py
This script contains functions to evaluate the models and the wikidata output.

### helper.py 
This script contains functions to merge checkpoint files and to examine how many dictionaries have been extracted from the received answers.

### LICENSE
The code is published open source under the [MIT License](https://opensource.org/license/mit).

### prompt.py
This script contains the class `Prompter` with functions to prompt the data samples and retrieve the generated output.

### query.py
This script contains functions to query the wikidata API and collect the information in the evaluation corpus.
