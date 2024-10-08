import os
from abc import ABC, abstractmethod
from typing import List, Tuple

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
    metrics: List[str]
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

       
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
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

        self.gradient_accumulation_steps = 4
        self.warmup_steps = 0
        self.train_batch_size = 8


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


    @staticmethod
    def split_dataset(dataset: Dataset, test_size: float, num_splits: int = 1, weights_train_test: List = None) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Split a dataset several times into a train and a test subdataset by computing two lists of indices for each split. If no weights are given, 
        the splittings are done using a uniform distribution. Weights can be send to the function as a list of numbers of the same size as the dataset. 
        Data with a high weight has more chance to be part of the training subdatasets.

        :param dataset: dataset to split.
        :param test_size: size of test dataset.
        :param num_splits: number of splits to do.
        :param weights_train_test: weights for each data of dataset, if not given a uniform distribution is used instead.

        :return: two lists of num_splits list(s), the first containing the training dataset(s) ids and the second the testing subdataset(s) ids.
        """

        train_ids = []
        test_ids = []
        if(weights_train_test != None):
            weights_train_test = np.array(weights_train_test)
            weights_train_test = 1-(np.exp(weights_train_test)/sum(np.exp(weights_train_test)))

        for _ in range(num_splits):
            shuffle_ids = np.random.choice(np.arange(len(dataset)), len(dataset), replace=False, p=weights_train_test)
            test_ids.append(shuffle_ids[:int(test_size*len(shuffle_ids))].tolist())
            train_ids.append(shuffle_ids[int(test_size*len(shuffle_ids)):].tolist())

        return train_ids, test_ids



    def run(self, dataset: Dataset, test_size: float, epochs: int, output_dir: Path, num_models: int = 1,\
     model_to_use : PreTrainedModel = None, output_name : str = "",  weights_train_test : List[int] = None,
     save_epochs: int = -1) -> (PreTrainedModel):
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

        tokenized_dataset = dataset.map(self._tokenize, batched=True, remove_columns=["text"])
        self.preprocessed_dataset = tokenized_dataset.map(self._preprocess_dataset, batched=True)

        if("raw_labels" in self.preprocessed_dataset.column_names):
            self.preprocessed_dataset = self.preprocessed_dataset.remove_columns("raw_labels")

        self.preprocessed_dataset.save_to_disk(output_dir / "preprocessed_dataset")

        if(save_epochs < 0):
            save_strategy="no"
            save_steps=1
        else:
            save_strategy = "steps"
            save_steps = (save_epochs*int(len(self.preprocessed_dataset)*(1-test_size)))//self.train_batch_size

        training_args = TrainingArguments(
            output_dir="checkpoints/"+output_dir.name, 
            seed=self.config.seed, 
            num_train_epochs=epochs, 
            save_strategy=save_strategy,
            save_steps=save_steps, 
            learning_rate=self.config.learning_rate,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_accumulation_steps=2,
            eval_strategy="steps",
            eval_steps=1000000,
            logging_dir=output_dir / "logs",
            logging_steps=100,
            remove_unused_columns=False,
            logging_first_step=True,
            fp16=True,
            warmup_steps=self.warmup_steps,
            weight_decay=0.1  # Augmentez le weight decay
        )

        tab_models = []
        tab_metrics = []

        if(weights_train_test != None):
            if(len(weights_train_test) != len(self.preprocessed_dataset)):
                print(f"WARNING: {len(weights_train_test)} weights for {len(self.preprocessed_dataset)} data, splitting train/test using uniform distribution!")
                weights_train_test = None

        self.train_datasets_ids, self.test_datasets_ids = Finetuner.split_dataset(self.preprocessed_dataset, test_size, num_models, weights_train_test)

        for i in range(num_models):

            train_index = self.train_datasets_ids[i]
            test_index = self.test_datasets_ids[i]

            ds = DatasetDict({"train": self.preprocessed_dataset.select(train_index), "test": self.preprocessed_dataset.select(test_index)})

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
            
            model_name_stripped = self.config.model_name.split("/")[-1]
            model_output = f"{model_name_stripped}_{output_name}{i}"
            print(f"Will save to {output_dir} / {model_output}")

            trainer.train()

            trainer.save_model(output_dir / model_output)
            print(f"Saved {output_dir}/{model_output}...")

            with open(output_dir / model_output / "ids.dict", "wb") as outfile:
                pickle.dump({"train": train_index, "test": test_index}, outfile)

            if(self.metrics != None):
                metric = trainer.evaluate()
                tab_metrics.append(metric)
                with open(output_dir / model_output / "metrics.dict", "wb") as outfile:
                    pickle.dump(metric, outfile)

            tab_models.append(model)

        return tab_models, tab_metrics
        

