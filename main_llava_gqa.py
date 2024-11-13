import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, StoppingCriteriaList
from PIL import Image
import pickle
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Define Dataset class for GQA
class GQADataset(Dataset):
    def __init__(self, config):
        self.root = config['root_dir']
        self.image_data_root = os.path.join(self.root, config['image_dir'])
        self.questions_data_root = os.path.join(self.root, config['question_file'])
        self.question_data = json.load(open(self.questions_data_root, 'r'))
        self.question_ids = list(self.question_data.keys())

    def __len__(self):
        return len(self.question_data.items())

    def __getitem__(self, idx):
        question_id = self.question_ids[idx]
        question = self.question_data[question_id]['question']
        promptified_question = f"Question: {question}"
        image_id = self.question_data[question_id]['imageId']
        return {
            'question_id': question_id,
            'promptified_question': promptified_question,
            'image_path': os.path.join(self.image_data_root, f'{image_id}.jpg')
        }

# Collate function to batch samples
def collate_fn(batch):
    question_ids = [sample['question_id'] for sample in batch]
    promptified_questions = [sample['promptified_question'] for sample in batch]
    image_paths = [sample['image_path'] for sample in batch]
    return {
        'question_ids': question_ids,
        'promptified_questions': promptified_questions,
        'image_paths': image_paths
    }

# Define custom stopping criteria
class CustomStoppingCriteria:
    def __init__(self, tokenizer, eos_sequences):
        self.tokenizer = tokenizer
        self.eos_sequences = eos_sequences

    def __call__(self, input_ids, scores):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for seq in self.eos_sequences:
            if seq in decoded_text:
                return True
        return False
    
# Generate explanations using LLaVA model
def generate_explanations_LLaVA(models, sample, config, config_logging):
    torch_device = models['llava']['device']
    model_llava = models['llava']['model']
    processor_llava = models['llava']['processor']
    model_llava.eval()

    prompt = sample['promptified_questions'][0]
    raw_image = Image.open(sample['image_paths'][0])
    question_id = sample['question_ids'][0]
    
    inputs = processor_llava(images=raw_image, text=prompt, return_tensors="pt").to(torch_device)
    explanation_path = os.path.join(config_logging['explanation_dir'], f"explanations_{question_id}.pkl")

    custom_eos_sequences = ['\n', '.']
    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(processor_llava.tokenizer, custom_eos_sequences)])
    if not os.path.exists(explanation_path):
        with torch.no_grad():
            outputs = model_llava.generate(
                input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], max_new_tokens=50, stopping_criteria=stopping_criteria
            )
            generated_token_ids = outputs.sequences[:, inputs['input_ids'].shape[-1]:] 
            decoded_output = processor_llava.decode(generated_token_ids[0], skip_special_tokens=True)

        sample['response'] = {
            'prompt': prompt,
            'decoded_output': decoded_output
        }
        with open(explanation_path, 'wb') as f:
            pickle.dump(sample, f)
        print(f'Explanations saved to {explanation_path}')

    return explanation_path

# Inference pipeline
def inference_pipeline(config):
    dataloader = DataLoader(
        GQADataset(config['data']),
        batch_size=1,
        collate_fn=collate_fn
    )

    model_id_llava = config['mm_model']['model_path']
    model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to('cuda')
    processor_llava = AutoProcessor.from_pretrained(model_id_llava)

    models = {'llava': {'model': model_llava, 'processor': processor_llava, 'device': 'cuda'}}
    for  sample in tqdm(dataloader, desc="Processing samples"):
        generate_explanations_LLaVA(models, sample, config['mm_model'], config['logging'])

# Main function to run the pipeline
if __name__ == "__main__":
    config = {
        'data': {
            'root_dir': '/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/',
            'image_dir': 'images',
            'question_file': 'questions1.2/train_all_questions/train_all_questions_0_random_filtered_10000.json'
        },
        'mm_model': {
            'model_path': 'meta-llama/Llama-2-7b-chat-hf',
        },
        'logging': {
            'explanation_dir': './explanations'
        }
    }
    inference_pipeline(config)