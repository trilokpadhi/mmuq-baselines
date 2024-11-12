import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
import numpy as np
from collections import Counter
from math import log2
import os
import pickle
import json

from rouge_score import rouge_scorer
import itertools
import math
from tqdm import tqdm

# Initialize the ROUGE scorer with ROUGE-L
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

nltk.download('punkt')

# Initialize models and tokenizers
deberta_model_name = 'microsoft/deberta-base-mnli'
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name)

# Load responses
# explanation_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/explanations'
explanation_dir = '/mnt/myebsvolume/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_10000/explanations'
# grounding_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/grounding'
grounding_dir = '/mnt/myebsvolume/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_10000/grounding'
# uncertainty_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty'
uncertainty_dir = '/home/ec2-user/Multimodal-Uncertainty-Quantification/runs/uncertainty'

# Method A: Log Probability of the sentence from the scores of the model
def get_log_prob_from_logits(logits, generated_token_ids):
    """
    logits: tuple of length of the no of tokens in the sentence, and each element is a tensor of size (1, vocab_size)
    generated_token_ids: tensor of size (1, no of tokens in the sentence) 
    """
    logits = torch.stack(logits, dim=1).squeeze(0)
    log_softmax = F.log_softmax(logits, dim=-1)
    generated_token_ids = generated_token_ids.view(-1, 1)
    log_softmax_of_generated_tokens = log_softmax.gather(1, generated_token_ids)
    log_prob_sentence = log_softmax_of_generated_tokens.sum().item()
    
    return log_prob_sentence
    
    
def calculated_predictive_entropy_from_log_probs(log_probs):
    """
    log_probs: list of log probabilities of the sentences
    """
    probabilities = [np.exp(lp) for lp in log_probs]
    probabilities = np.array(probabilities)
    log_probs = np.array(log_probs)
    
    total_predicitive_entropy = np.sum(probabilities*log_probs)
    
    return total_predicitive_entropy
    
def get_json_from_response(response):
    """
    response is usually a string which has lot of extrac characters instead of the json, like as below:
    '1 {\n      "answer": "No",\n      "explanation": "It is only the food on the plate",\n      "confidence": 1\n    }'
    """
    start_token = response.find('{')
    end_token = response.rfind('}')
    response_json_str = response[start_token:end_token+1]
    response_json = json.loads(response_json_str)
    return response_json
    
def get_lexical_similarity(responses):
    """
    get rouge-l score between the two sentences
    """
    rouge_l_scores = []
    counter = 0 
    # Generate all unique pairs
    for resp1, resp2 in itertools.combinations(responses, 2):
        score = scorer.score(resp1, resp2)
        rouge_l_f1 = score['rougeL'].fmeasure  # Get F1 score for ROUGE-L
        rouge_l_scores.append(rouge_l_f1)
        counter += 1
        
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)        
    return average_rouge_l
    
def get_accuracy(full_answers, questions, responses):
    """
    full_answers: list of full answers
    questions: list of questions
    responses: list of responses
    and calculate the accuracy by entailment of the responses with the full answers
    """
    correct = 0
    for response in responses:
        premise = f"{questions} {full_answers}"
        hypthesis = f"{questions} {response}"
        
        inputs = deberta_tokenizer.encode_plus(premise, hypthesis, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = deberta_model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            if pred == 2:
                correct += 1
    accuracy = correct / len(responses)
        
    return accuracy

def get_semantic_entropy(question, explanation_with_log_probs):
    """
    Semantic entropy is calculated by first forming clusters from the sentences based on the bidirectional entailment scores. 
    Then the entropy is calculated using the clusters formed.
    Semantic entropy is calculated as the entropy of the clusters formed.
    Semantic entropy is calculated as the entropy of the clusters formed, Probability of a cluster is the average of the probabilities of the sentences in the cluster. 
    question is the question for which the explanations are generated
    explanation_with_log_probs: dictionary with keys as the explanations and values as the log probabilities of the
    
    ### Pseudo code
    Calculates the semantic entropy for a given question and its explanations.

    Semantic entropy is computed by:
    1. Clustering explanations based on bidirectional entailment using an NLI model.
    2. Calculating the entropy over the distribution of these clusters.

    Args:
        question (str): The question for which the explanations are generated.
        explanation_with_log_probs (dict): A dictionary where keys are explanations (str)
                                           and values are their corresponding log probabilities (float).

    Returns:
        float: The semantic entropy value.
    """
    # Extract explanations and their log probabilities
    explanations = list(explanation_with_log_probs.keys())
    log_probs = list(explanation_with_log_probs.values())

    # Convert log probabilities to probabilities
    probs = [math.exp(lp) for lp in log_probs]

    # Normalize probabilities to ensure they sum to 1
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    # Initialize clusters
    clusters = []

    for idx, explanation in enumerate(explanations):
        assigned = False
        for cluster in clusters:
            representative = cluster[0]  # Choose the first explanation as the representative

            # Prepare premises for NLI
            premise1 = f"{question} {explanation}"
            premise2 = f"{question} {representative}"

            # Encode the pair for NLI: premise -> hypothesis
            inputs1 = deberta_tokenizer.encode_plus(premise1, premise2, return_tensors='pt', truncation=True)
            inputs2 = deberta_tokenizer.encode_plus(premise2, premise1, return_tensors='pt', truncation=True)

            with torch.no_grad():
                logits1 = deberta_model(**inputs1).logits
                logits2 = deberta_model(**inputs2).logits

            # Get predicted labels
            pred1 = torch.argmax(logits1, dim=1).item()  # 0: contradiction, 1: neutral, 2: entailment
            pred2 = torch.argmax(logits2, dim=1).item()

            # Check for bidirectional entailment
            if pred1 == 2 and pred2 == 2:
                cluster.append(explanation)
                assigned = True
                break  # Move to the next explanation after assigning to a cluster

        if not assigned:
            # Create a new cluster with the current explanation as its representative
            clusters.append([explanation])

    # Compute the probability of each cluster by summing probabilities of its explanations
    cluster_probs = []
    for cluster in clusters:
        # cluster_prob = sum(probs[explanations.index(exp)] for exp in cluster)
        cluster_prob = np.mean([probs[explanations.index(exp)] for exp in cluster])
        cluster_probs.append(cluster_prob)

    # Compute semantic entropy: SE(x) = -sum(p(c) * log(p(c)))
    entropy = -sum(p * math.log(p) for p in cluster_probs if p > 0)
    num_clusters = len(clusters)

    return entropy, num_clusters
    
    
    
# Get the responses from the files and store them in a dictionary
responses = {}
files = os.listdir(explanation_dir)
for file in tqdm(files, desc='Processing files', total=len(files)):
    # Get probability of the 20 responses
    uncertainty_scores_baseline = {}
    # predictive_entropy = {}
    # lexical_similarity = {}
    # semantic_entropy = {}
    with open(os.path.join(explanation_dir, file), 'rb') as f:
        question_id = file.split('_')[1].split('.')[0] # since file name is of the format - explanations_181060668.pkl
        explanation_metadata = pickle.load(f)
        log_probs = []
        rogue_scores = []
        explanations = []
        explanation_with_log_probs = {}
        question = explanation_metadata['questions'][0]
        for key, value in explanation_metadata.items():
            if 'response' in key:
                metadata = explanation_metadata[key]
                # logits = metadata['outputs'].scores
                # logits = metadata['transition_scores']
                # generated_token_ids = metadata['generated_token_ids']
                # log_prob_sentence = get_log_prob_from_logits(logits, generated_token_ids)
                log_prob_sentence = np.sum(metadata['transition_scores'])
                log_probs.append(log_prob_sentence)
                
                ## get explanations from the model response dict to calculate rogue scores
                try:
                    response_json = get_json_from_response(metadata['decoded_outputs'])
                    explanation = response_json['explanation']
                    explanations.append(explanation)
                    
                    # for semantic entropy
                    # explanation_with_log_probs[key] = {}
                    # explanation_with_log_probs[key]['explanation'] = explanation
                    # explanation_with_log_probs[key]['log_prob'] = log_prob_sentence
                    explanation_with_log_probs[explanation] = log_prob_sentence
                    
                except Exception as e:
                    # print(f"Error in getting explanation from response: {metadata['decoded_outputs']}")
                    print(f"Error: {e}")
                    continue
                
            else:
                continue
        predictive_entropy = calculated_predictive_entropy_from_log_probs(log_probs)
        rogue_l_score = get_lexical_similarity(explanations)
        semantic_entropy, num_clusters = get_semantic_entropy(question, explanation_with_log_probs)
        explanation_ = list(explanation_with_log_probs.keys())
        accuracy = get_accuracy(explanation_metadata['full_answers'][0], explanation_metadata['questions'][0], explanation_)
        uncertainty_scores_baseline = {}
        uncertainty_scores_baseline['predictive_entropy'] = predictive_entropy
        uncertainty_scores_baseline['lexical_similarity'] = rogue_l_score
        uncertainty_scores_baseline['semantic_entropy'] = semantic_entropy
        uncertainty_scores_baseline['num_clusters'] = num_clusters
        uncertainty_scores_baseline['accuracy'] = accuracy
        
    responses[question_id] = uncertainty_scores_baseline
    

# make the directory if it does not exist
if not os.path.exists(uncertainty_dir):
    os.makedirs(uncertainty_dir)
# Save the responses in a pickle file
with open(os.path.join(uncertainty_dir, 'uncertainty_scores_baseline.pkl'), 'wb') as f:
    pickle.dump(responses, f)
    

        
                
                 