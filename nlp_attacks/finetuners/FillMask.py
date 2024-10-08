from typing import Dict
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForWholeWordMask, EvalPrediction, AutoModelForMaskedLM
import xml.etree.ElementTree as ET
import jsonlines
import numpy as np
from pathlib import Path

from .Finetuner import Finetuner, FinetunerConfig

class FillMask(Finetuner):
    def __init__(self, config: FinetunerConfig) -> None:
        super().__init__(config)

        self.data_collator = DataCollatorForWholeWordMask(self.tokenizer, mlm_probability=0.15)

    def _preprocess_dataset(self, examples: Dataset) -> Dataset:
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than self.context_length
        total_length = (total_length // self.config.context_length) * self.config.context_length
        # Split by chunks of max_len
        result = {
            k: [t[i : i + self.config.context_length] for i in range(0, total_length, self.config.context_length)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result


    def _tokenize(self, examples: Dataset) -> Dataset:
        """Tokenize parts of a dataset without padding or truncation because 
        a batching is applied afterwards through the _preprocess_dataset function.
        This function is meant to be used with the map function of a Dataset.

        :param examples: part of the dataset to tokenize

        :return: new dataset with texts tokenized and then decoded.
        """
        return self.tokenizer(examples["text"])

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for training.
        :param eval_prediction: contains the raw predictions and the label ids.
        :return: metrics results.
        """
        raw_predictions, label_ids = eval_prediction
        predictions = np.argmax(raw_predictions, axis=-1)

        # Flatten the arrays
        predictions = predictions.flatten()
        label_ids = label_ids.flatten()

        # Filter out masked tokens (-100)
        mask = label_ids != -100
        filtered_predictions = predictions[mask]
        filtered_labels = label_ids[mask]

        # Compute metrics
        metrics_result = self.metrics.compute(predictions=filtered_predictions, references=filtered_labels)

        return metrics_result
    
    def _load_model(self):
        return AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        
