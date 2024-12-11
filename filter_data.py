import json
import random

def load_json_file(file_path):
    """Load JSON data from the given file path."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json_file(data, file_path):
    """Save JSON data to the given file path."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def sample_data(data, sample_size):
    """Randomly sample data from the dataset."""
    if isinstance(data, list):
        if len(data) <= sample_size:
            return data
        return random.sample(data, sample_size)
    elif isinstance(data, dict):
        if len(data) <= sample_size:
            return data
        return random.sample(list(data.items()), sample_size)
    else:
        raise ValueError("Unsupported data type. Data should be a list or a dictionary.")

def main():
    
    """
    for GQA dataset
    """
    questions_file = '/mnt/data/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
    sample_size = 10  # Change this value to sample a different number of questions
    output_file = 'test.json'
    

    # Load dataset
    dataset = load_json_file(questions_file) # for GQA dataset

    # Sample the dataset
    sampled_data = sample_data(dataset, sample_size)

    # Convert the sampled data back to dictionary (from list of tuples)
    sampled_dict = dict(sampled_data)

    # Save the sampled dataset
    save_json_file(sampled_dict, output_file) # for GQA dataset

    print(f"Sampled {sample_size} entries and saved to {output_file}")

if __name__ == '__main__':
    main()