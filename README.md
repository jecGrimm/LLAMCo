# LLAMCo (LLMs As Metadata Collectors)
This repository test whether Large Language Models (LLMs) can be used as metadata collectors. We use an exisiting metadata corpus for German 19th century literature and test whether LLMs are able to collect the same metadata information via prompting strategies.

## Installation
1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/index.html).
2. Create environment:<br> `conda env create -f env.yml`
3. Activate environment:<br> `conda activate llamco`

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
