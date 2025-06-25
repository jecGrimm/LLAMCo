# LLAMCo (LLMs As Metadata Collectors)
This repository test whether Large Language Models (LLMs) can be used as metadata collectors. We use an exisiting metadata corpus for German 19th century literature and test whether LLMs are able to collect the same metadata information via prompting strategies.

## Installation
1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/index.html).
2. Create environment:<br> `conda env create -f env.yml`
3. Activate environment:<br> `conda activate llamco`

## Usage
### Slurm
#### Commands
`sbatch --partition=NvidiaAll ./scripts/prompt.sh`

#### Jobs
prompt (test Llama): 
- 11035: failed because of missing output directory (possibly)
- 11037 (output dir kreiert): failed weil das Repo falsche benannt war
- 11050 (LLAMCO in LLAMCo umbenannt): failed weil das Modell nicht geladen werden konnte
- 11456 (mit accelerate installed): failed, weil sample in generate_text nicht übergeben wurde
- 11457 (lamdba function in generate): cancelled, um das ganze Datenset laufen zu lassen
- 11458 (pipe mit ganzem Datenset): failed, weil pipe ein Dataset übergeben bekommen hat
- 11460 (pipe mit List): failed, weil Liste keine Funktion tojson hat
- 12492 (save list as normal json): failed, weil die directories schon bestanden 
- 13191 (fixed existing output dir, data only first 3): failed, weil prompt_samples ein dict ist
- 13192 (fixed huggingface issue): failed, weil indexing dir zurückgibt
- 13193 (select für testindices): failed, weil ich das exist_ok nicht gepullt habe
- 13194 (exist_ok = True gesetzt): failed, weil output json keine Datei ist
- 13195 (model_id in json gesplittet): failed, weil select mit falschen Klammern benutzt wurde
- 13196 (fixed select): failed, weil man nicht über einen int iterieren kann in select
- 13198 (fixed select list): COMPLETED
- 14941 (Llama 8B): failed, weil env nicht aktiviert war
- 14942 (Llama 8B mit llamco-env): failed, weil hf down ist
- 14943 (hf langsam, aber wieder da): failed, weil hf immer noch down ist
- 15002 (hf wieder up): failed, OOM

## Structure
### data
This directory contains the hugginface datasets.
#### eval_samples
This directory contains the huggingface dataset with the evaluation samples.
#### prompt_samples
This directory contains the huggingface dataset with the prompt samples.

### notebooks
This directory contains colab notebooks.
#### explore_Llama.ipynb
Notebook prompting Llama.

### resources
This directory contains resources deployed for the experiments.
#### Bibliographie.tsv
Metadata corpus used as the foundation of the experiments.

### data.py
Script that reads the data from the metadata corpus, transforms it into prompt and evaluation samples. It also stores or loads the already stored samples as a huggingface dataset from or in the folder `data`.

### env.yml
Environment file for a quick installation of the needed dependencies via conda. See Section `Installation` for a guide how to create a new environment with this file.

### explore_llama.py
This script is the transformed version of explore_Llama.ipynb and contains a script to prompt Llama-3.2-1B-Instruct.

### LICENSE
The code is published open source under the [MIT License](https://opensource.org/license/mit).

### prompt.py
This script contains functions to prompt the data samples and retrieve the generated output.
