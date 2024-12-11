import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, StoppingCriteriaList
from PIL import Image
import pickle
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import json
import numpy as np

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
        self.answer = self.question_data[question_id]['answer']
        self.full_answer = self.question_data[question_id]['fullAnswer']
        # promptified_question = f"Question: {question}"
        promptified_question = f"USER: <image> {question}, ASSISTANT:"
        image_id = self.question_data[question_id]['imageId']
        return {
            'question': question,
            'answer': self.answer,
            'full_answer': self.full_answer,
            'question_id': question_id,
            'promptified_question': promptified_question,
            'image_path': os.path.join(self.image_data_root, f'{image_id}.jpg')
        }

# Collate function to batch samples
def collate_fn(batch):
    questions = [sample['question'] for sample in batch]
    answers = [sample['answer'] for sample in batch]
    full_answers = [sample['full_answer'] for sample in batch]
    question_ids = [sample['question_id'] for sample in batch]
    promptified_questions = [sample['promptified_question'] for sample in batch]
    image_paths = [sample['image_path'] for sample in batch]
    return {
        'questions': questions,
        'answers': answers,
        'full_answers': full_answers,
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
def generate_explanations_LLaVA(models, sample, config):
    torch_device = models['llava']['device']
    model_llava = models['llava']['model']
    processor_llava = models['llava']['processor']
    model_llava.eval()

    prompt = sample['promptified_questions'][0]
    raw_image = Image.open(sample['image_paths'][0])
    question_id = sample['question_ids'][0]
    
    inputs = processor_llava(images=raw_image, text=prompt, return_tensors="pt").to(torch_device)
    explanation_path = os.path.join(config['logging']['explanation_dir'], f"explanations_{question_id}.pkl")
    if os.path.exists(explanation_path):
        print(f"Explanations already generated for question {question_id}")
        return explanation_path
    else:
        os.makedirs(config['logging']['explanation_dir'], exist_ok=True)
        # custom_eos_sequences = ['\n', '.']
        # stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(processor_llava.tokenizer, custom_eos_sequences)])
        decoded_output_list = []
        transition_scores_list = []
        probabilties_list = []
        # if not os.path.exists(explanation_path):
        print(f"Generating explanations for question {question_id}")
        # for i in range(20):
        for i in range(config['no_of_responses_sampled_per_image']):
            print(f"Generating explanation {i+1} of {config['no_of_responses_sampled_per_image']}")
            outputs = model_llava.generate(
                input_ids=inputs['input_ids'],        # Text tokens
                pixel_values=inputs['pixel_values'],  # Image tokens
                do_sample=True,                      # Enable sampling
                temperature=config['temperature'],    # Sampling temperature
                top_p=config['top_p'],                # Top-p sampling
                num_beams=config['num_beams'],        # Beam search
                max_new_tokens=config['max_new_tokens'],
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=True,
                # stopping_criteria=stopping_criteria
            )
            
            # Process the generated output, remove the input token ids 
            generated_token_ids = outputs.sequences[:, inputs['input_ids'].shape[-1]:] 
            decoded_output = processor_llava.decode(generated_token_ids.flatten(), skip_special_tokens=True)
            # decoded_output_list.append(decoded_output)
            # Compute transition probabilities
            transition_score = model_llava.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).tolist()
            probabilties = np.exp(transition_score)
            # transition_scores_list.append(transition_score) # transition scores for each token in the output sequence are log probabilities
            # probabilties_list.append(probabilties) # Also store the probabilities for each token in the output sequence
        
            # log for each response
            sample[f'response_{i}'] = {
                'prompt': prompt,
                'decoded_output': decoded_output,
                'transition_scores': transition_score,
                'probabilities': probabilties
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
    i = 0
    for sample in tqdm(dataloader, desc="Processing samples"):
        i += 1
        if config['debug']:
            if i > 1:
                break
        generate_explanations_LLaVA(models, sample, config)

# Main function to run the pipeline
if __name__ == "__main__":
    config = {
        'data': {
            'root_dir': '/mnt/data/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/',
            'image_dir': 'images',
            'question_file': 'questions1.2/train_all_questions/test.json'
        },
        'mm_model': {
            'model_path': "llava-hf/llava-1.5-7b-hf",
        },
        'logging': {
            'explanation_dir': 'explanations'
        },
        'no_of_responses_sampled_per_image': 5,
        'temperature': 0.7,
        'top_p': 0.9,
        'num_beams': 1,
        'max_new_tokens': 100,
        'debug': False
    }
    inference_pipeline(config)
    print("Inference pipeline completed")