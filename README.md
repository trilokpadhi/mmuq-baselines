# MMUQ Baselines
- `run_baseline.py` file with the code for the baseline models, The baseline models are:
    - Predictive Entropy
    - Semantic Entropy
    - Semantic Clusters (No. of clusters formed from the responses)
    - Lexical Similarity 

- `token-probs.py` file with the code for the token probabilities of the response from `model.generate()`
- `llava-inference.ipynb` file with the code for the inference of the LLaVA 1.5 model

# Datasets
- Download the GQA dataset from [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)

# Steps to Run 
- After downloading the datasets, please set the path for the dataset in [main_llava_gqa.py](https://github.com/trilokpadhi/mmuq-baselines/blob/main/main_llava_gqa.py)
- This will log all the generations and the artifacts in the `runs/` directory, with explanations.
- After this please update the paths of these explanations in [run_baselines.py](https://github.com/trilokpadhi/mmuq-baselines/blob/main/run_baselines.py)
