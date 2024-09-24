import os
from abc import ABC, abstractmethod

import numpy as np
import random
import torch
import evaluate
import pickle
import copy
from datasets import Dataset, DatasetDict
from sklearn.model_selection import ShuffleSplit
from transformers import AutoTokenizer, TrainingArguments, PreTrainedModel, Trainer, EvalPrediction, PreTrainedTokenizerFast
from pathlib import Path
from dataclasses import dataclass
import warnings

@dataclass
class FinetunerConfig:
    """
    Configuration class for the fine-tuning process.

    Attributes:
        model_name (str): The name of the pre-trained model to be fine-tuned.
        seed (int): Random seed to ensure reproducibility.
        context_length (int): Length of the context window for tokenization.
        metrics (list[str]): List of metrics to use for evaluation.
        learning_rate (float): The learning rate for training.
    """
    model_name: str
    seed: int
    context_length: int
    metrics: list[str]
    learning_rate: float


class Finetuner(ABC):
    """
    Base class for fine-tuning machine learning models.

    Attributes:
        config (FinetunerConfig): Configuration object containing model settings.
        metrics (Optional[Callable]): Combined evaluation metrics function.
        tokenizer (AutoTokenizer): Tokenizer object for the model.
        train_datasets_ids (list): List of training dataset indices.
        test_datasets_ids (list): List of test dataset indices.
        data_collator (Optional[Callable]): Data collator for batching.
    """
    def __init__(self, config: FinetunerConfig) -> None:
        self.config = config
        
        if(len(config.metrics) > 0):
            self.metrics = evaluate.combine(config.metrics)
        else:
            self.metrics = None

        # We want fast tokenizers if possible and print a warning if we can't have one
        if config.tokenizer_name is None:
            print("No tokenizer name provided, using model name")
            config.tokenizer_name = config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        if not isinstance(self.tokenizer, (PreTrainedTokenizerFast)):
            warnings.warn("Using a slow tokenizer, consider using a fast tokenizer to have faster training")
        else:
            print("Using a fast tokenizer")
        if("gpt" in self.config.model_name):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
             
        self.train_datasets_ids = []
        self.test_datasets_ids = []
        self.data_collator = None


    def _compute_metrics(self, eval_prediction: EvalPrediction):
        """
        Computes metrics for model evaluation. This method must be implemented in subclasses.

        Args:
            eval_prediction (EvalPrediction): Predictions and labels for evaluation.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Must be implemented in subclasses")

    def _reset_seeds(self) -> None:
        """
        Resets random seeds for reproducibility across numpy, random, and torch.
        """
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)


    def run(self, dataset: Dataset, test_size: float, epochs: int, output_dir: Path, num_models: int = 1,\
     model_to_use : PreTrainedModel = None, output_name : str = "") -> (PreTrainedModel):
        """
        Executes the fine-tuning process.

        Args:
            dataset (Dataset): The dataset to use for training and evaluation.
            test_size (float): Proportion of the dataset to use for testing.
            epochs (int): Number of training epochs.
            output_dir (Path): Directory to save models and logs.
            num_models (int, optional): Number of models to fine-tune with different splits. Defaults to 1.
            model_to_use (PreTrainedModel, optional): Already finetuned model to use. Defaults to None (a pre-trained only model is then used).
            output_name (str, optional): Output name prefix for the models. Defaults to "".

        Returns:
            list[PreTrainedModel], list[dict]: A list of fine-tuned models and corresponding evaluation metrics.
        """

        self._reset_seeds()

        print(dataset)
        tokenized_dataset = dataset.map(self._tokenize, batched=True, remove_columns=["text"])
        print(tokenized_dataset)
        self.preprocessed_dataset = tokenized_dataset.map(self._preprocess_dataset, batched=True)
        print(self.preprocessed_dataset)

        if("raw_labels" in self.preprocessed_dataset.column_names):
            self.preprocessed_dataset = self.preprocessed_dataset.remove_columns("raw_labels")

        self.preprocessed_dataset.save_to_disk(output_dir / "preprocessed_dataset")

        training_args = TrainingArguments(
            output_dir="checkpoints/"+output_dir.name, 
            seed=self.config.seed, 
            num_train_epochs=epochs, 
            save_steps=25000, 
            learning_rate=self.config.learning_rate,
            per_device_eval_batch_size=32,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=2,
            eval_strategy="steps",
            eval_steps=1000000,
            logging_dir=output_dir / "logs",
            logging_steps=100,
            remove_unused_columns=False,
            logging_first_step=True,
            fp16=True,
            weight_decay=0.1  # Augmentez le weight decay
        )

        tab_models = []
        tab_metrics = []

        rs = ShuffleSplit(n_splits=num_models, test_size=test_size, random_state=self.config.seed)

        for i, (train_index, test_index) in enumerate(rs.split(self.preprocessed_dataset)):
            ds = DatasetDict({"train": self.preprocessed_dataset.select(train_index), "test": self.preprocessed_dataset.select(test_index)})

            self.train_datasets_ids.append(train_index)
            self.test_datasets_ids.append(test_index)

            if(model_to_use == None):
                model = self._load_model()
            else:
                model = model_to_use

            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using {torch.cuda.get_device_name()}")
                model = model.to(device)
                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs")
                    torch.nn.DataParallel(model)
            else :
                print("GPU not available, using CPU")
                device = torch.device("cpu")
                model = model.to(device)
            
            trainer = Trainer(
                model=model, 
                args=training_args, 
                train_dataset=ds["train"], 
                eval_dataset=ds["test"],
                data_collator=self.data_collator, 
                compute_metrics=self._compute_metrics
            )

            trainer.train()

            model_output = f"{self.config.model_name}_{output_name}{i}"
            trainer.save_model(output_dir / model_output)

            with open(output_dir / model_output / "ids.dict", "wb") as outfile:
                pickle.dump({"train": train_index, "test": test_index}, outfile)

            if(self.metrics != None):
                metric = trainer.evaluate()
                tab_metrics.append(metric)
                with open(output_dir / model_output / "metrics.dict", "wb") as outfile:
                    pickle.dump(metric, outfile)

            tab_models.append(model)

        return tab_models, tab_metrics
        

