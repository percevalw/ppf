import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import nlp_attacks.preprocessing
from torch.nn.parallel import DataParallel
import random
import json
import os

from ..utils import group_texts, tokenize_function

@dataclass
class MLMEvaluationConfig:
    """
    Configuration class for Masked Language Model (MLM) evaluation.

    Attributes:
        model_names (List[str]): List of model names to be evaluated.
        model_paths (List[str]): List of paths to pretrained models.
        tokenizer_path (str): Path to the tokenizer to be used.
        output_dir (str): Directory where the results will be saved.
        mode (str): Mode used to generate words_to_mask (e.g., 'direct', 'indirect', 'all').
        words_not_to_mask (set): Set of words that should not be masked during evaluation.
        mask_ratio (float): Percentage of tokens to be masked. Default is 0.15.
        batch_size (int): Batch size for the DataLoader. Default is 32.
    """
    model_names: List[str]
    model_paths: List[str]
    tokenizer_path: str
    output_dir: str
    mode: str
    words_not_to_mask: set()
    mask_ratio: float = 0.15
    batch_size: int = 32
    context_length: int = 128

@dataclass
class MLMEvaluationRunParameters:
    """
    Class for specifying run parameters for the MLM evaluation.

    Attributes:
        num_examples_to_print (int): Number of examples to print for detailed analysis (not useful for now). Default is 5. 
    """
    num_examples_to_print: int = 5

class MLMEvaluation:
    """
    Class for running Masked Language Model (MLM) evaluation on a given dataset.

    Attributes:
        dataset (Dataset): The dataset to be evaluated, it need to have a text column.
        config (MLMEvaluationConfig): Configuration for the evaluation.
        tokenizer (AutoTokenizer): Tokenizer for text preprocessing.
        models (List[AutoModelForMaskedLM]): List of pretrained MLM models.
        output_file (str): Path to the output file for saving results.
    """
    def __init__(self, dataset: Dataset, config: MLMEvaluationConfig):
        """
        Initializes the MLM evaluation with the dataset and configuration.

        Args:
            dataset (Dataset): The dataset for evaluation.
            config (MLMEvaluationConfig): Configuration for evaluation.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.models = [AutoModelForMaskedLM.from_pretrained(path) for path in config.model_paths]

        self.output_file = os.path.join(self.config.output_dir, f"{self.config.tokenizer_path}_{self.config.mode}_results.json")

        self.dataset = dataset

    
    def get_whole_word_mask_indices(self, tokens: List[str]) -> List[int]:
        """
        Identifies word indices in a sequence that should be masked with respect to words not to mask.

        Args:
            tokens (List[str]): List of tokenized words.
            mask_ratio (float): Ratio of tokens to mask. Default is 0.15.

        Returns:
            List[int]: List of indices for the tokens to be masked.
        """
        special_tokens = {self.tokenizer.cls_token, self.tokenizer.sep_token}
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token in special_tokens:
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        
        random.shuffle(cand_indexes)
        num_to_mask = max(1, int(round(len(tokens) * self.config.mask_ratio)))
        covered_indexes = set()

        for index_set in cand_indexes:
            if len(covered_indexes) >= num_to_mask:
                break
            
            token_sequence = [tokens[i] for i in index_set]
            whole_word = self.tokenizer.convert_tokens_to_string(token_sequence)
            
            if whole_word in self.config.words_not_to_mask:
                continue
            
            if len(covered_indexes) + len(index_set) > num_to_mask:
                continue
            
            covered_indexes.update(index_set)

        return list(covered_indexes)

    def decode_predictions(self, input_ids: torch.Tensor, masked_input_ids: torch.Tensor, predictions: torch.Tensor, 
        tokenizer: AutoTokenizer) -> List[dict]:
        """
        Decodes predictions made by the model into human-readable text.

        Args:
            input_ids (torch.Tensor): Original input IDs.
            masked_input_ids (torch.Tensor): Masked input IDs with [MASK] tokens.
            predictions (torch.Tensor): Predictions made by the model.
            tokenizer (AutoTokenizer): Tokenizer used for decoding.

        Returns:
            List[dict]: List of dictionaries containing original, masked, and predicted text.
        """
        decoded = []
        for i, (inp, masked, pred) in enumerate(zip(input_ids, masked_input_ids, predictions)):
            original = tokenizer.decode(inp)
            masked_text = tokenizer.decode(masked)
            predicted = inp.clone()
            predicted[masked != inp] = pred[masked != inp]
            predicted_text = tokenizer.decode(predicted)
            decoded.append({
                'original': original,
                'masked': masked_text,
                'predicted': predicted_text
            })
        return decoded

    def create_and_evaluate_test_set(self, dataset: Dataset, num_examples_to_print: int = 5) -> Tuple[List[float], List[float]]:
        """
        Runs MLM evaluation on a given dataset and calculates accuracy.

        Args:
            dataset (Dataset): Tokenized dataset for evaluation.
            num_examples_to_print (int): Number of examples to print for analysis. Default is 5.

        Returns:
            Tuple[List[float], List[float]]: List of accuracy and top-5 accuracy for each model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models:
            model.to(device)
            if torch.cuda.device_count() > 1:
                model = DataParallel(model)

        total_correct = [0 for _ in range(len(self.models))]
        top5_total_correct = [0 for _ in range(len(self.models))]
        total_predictions = 0

        progress_bar = tqdm(total=len(dataset), desc="Evaluating", unit="example")

        for batch_idx, batch in enumerate(dataset.iter(batch_size=self.config.batch_size)):
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            masked_input_ids = torch.tensor(batch['masked_input_ids']).to(device)
            mask_arr = torch.tensor(batch['mask_arr']).to(device)

            with torch.no_grad():
                for model in self.models:
                    outputs = model(masked_input_ids, attention_mask=attention_mask).logits
                    predictions = outputs[mask_arr].argmax(dim=-1)
                    correct = (predictions == input_ids[mask_arr]).sum().item()
                    total_correct[self.models.index(model)] += correct

                    top5_predictions = outputs[mask_arr].topk(5, dim=-1).indices
                    top5correct = (top5_predictions == input_ids[mask_arr].unsqueeze(-1)).any(dim=-1).sum().item()
                    top5_total_correct[self.models.index(model)] += top5correct

            total_predictions += mask_arr.sum().item()
            
            progress_bar.update(input_ids.size(0))
        
        progress_bar.close()

        accuracy = [total_correct[i] / total_predictions for i in range(len(self.models))]
        top5_accuracy = [top5_total_correct[i] / total_predictions for i in range(len(self.models))]

        return accuracy, top5_accuracy

    def tokenize_and_batch(self, dataset: Dataset) -> Dataset:
        """
        Tokenizes and batches a dataset.

        Args:
            dataset (Dataset): Dataset containing the text data.

        Returns:
            Dataset: Tokenized and batched dataset.
        """
        data = pd.DataFrame({"text": dataset["text"]})
        dataset = Dataset.from_pandas(data)

        print(dataset)


        # On tokenize
        tokenized_datasets = dataset.map(lambda x: tokenize_function(x, self.tokenizer), 
            batched=True, 
            remove_columns=dataset.column_names,
            batch_size=128
        )

        print(tokenized_datasets)



        # Créer des morceaux de texte
        lm_datasets = tokenized_datasets.map(lambda x: group_texts(x, self.config.context_length),
            batched=True,
            batch_size=128
        )

        print(lm_datasets)

        return lm_datasets

    def mask_tokens(self, example: dict) -> dict:
        """
        Masks the tokenized data with respect to the words not to mask.

        Args:
            example (dict): Input example containing tokenized data.

        Returns:
            dict: Preprocessed example with input IDs, attention mask, masked input IDs, and mask array.
        """

        input_ids = torch.tensor(example['input_ids'])
        attention_mask = torch.tensor(example['attention_mask'])

        masked_input_ids = input_ids.clone()
        mask_arr = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, sentence in enumerate(input_ids):
            tokens = self.tokenizer.convert_ids_to_tokens(sentence.tolist())
            mask_indices = self.get_whole_word_mask_indices(tokens)
            
            for idx in mask_indices:
                masked_input_ids[i, idx] = self.tokenizer.mask_token_id
                mask_arr[i, idx] = True

        return {
            'input_ids': input_ids.tolist(),
            'attention_mask': attention_mask.tolist(),
            'masked_input_ids': masked_input_ids.tolist(),
            'mask_arr': mask_arr.tolist()
        }



    def run(self, run_params: MLMEvaluationRunParameters) -> dict:
        """
        Executes the full MLM evaluation process and saves the results.

        Args:
            run_params (MLMEvaluationRunParameters): Parameters for running the evaluation.

        Returns:
            dict: Dictionary containing accuracy and top-5 accuracy results.
        """

        lm_datasets = self.tokenize_and_batch(self.dataset)

        preprocessed_data = lm_datasets.map(self.mask_tokens, batched=True,
            batch_size=self.config.batch_size
        )

        accuracy, top5_accuracy = self.create_and_evaluate_test_set(preprocessed_data, run_params.num_examples_to_print)

        dict_accuracy = {mn: acc for mn, acc in zip(self.config.model_names, accuracy)}
        dict_top5_accuracy = {mn: acc for mn, acc in zip(self.config.model_names, top5_accuracy)}
        
        os.makedirs(self.config.output_dir, exist_ok=True)


        with open(self.output_file, 'w') as f:
            json.dump({
                'accuracy': dict_accuracy,
                'top5_accuracy': dict_top5_accuracy
            }, f)
        
        print(f"Results saved to {self.output_file}")
        
        return {
            'accuracy': dict_accuracy,
            'top5_accuracy': dict_top5_accuracy
        }