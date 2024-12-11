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

1. Create a filtered dataset:
    ```bash
    python filter_data.py --questions_file <path_to_questions_file> --sample_size <number_of_questions_to_sample> --output_file <output_file_path>
    ```

2. Set the dataset paths in [main_llava_gqa.py](https://github.com/trilokpadhi/mmuq-baselines/blob/main/main_llava_gqa.py):
    - `root_dir`: Path to the root directory of the dataset
    - `questions_file`: Path to the questions file relative to the root directory
    - `image_dir`: Path to the images directory relative to the root directory

3. The generations and artifacts will be logged in the `runs/` directory with explanations.

4. Update the paths of these explanations in [run_baselines.py](https://github.com/trilokpadhi/mmuq-baselines/blob/main/run_baselines.py).
