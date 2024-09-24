# Made by helain zimmermann following what has been done previously by DABADIE Hugo, MAGNANA Lucas and BERTHELIER Gaspard

import pandas as pd
import numpy as np
import random
import warnings
import torch
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from typing import List, Union, Any, Dict, Optional, Tuple
from collections.abc import Mapping
import collections

from transformers import (
    DataCollatorForWholeWordMask, 
    BertTokenizer, 
    BertTokenizerFast, 
    PreTrainedTokenizerBase,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    default_data_collator,
    AutoModelForMaskedLM
)


from .Finetuner import Finetuner, FinetunerConfig
from .FillMask import FillMask

@dataclass
class General_Special_FillMaskConfig(FinetunerConfig):
    """
    Configuration class for the General_Special_FillMask fine-tuning process.

    Attributes:
        exclude_words (list[str], optional): List of words to exclude from masking.
        mlm_probability (float): Probability of masking tokens for MLM.
        tokenizer_name (str): The name of the tokenizer to use.
    """
    
    exclude_words: list[str] = None
    mlm_probability: float = 0.15
    tokenizer_name: str = "bert-base-cased"


class General_Special_FillMask(FillMask):
    """
    Fine-tuning class for applying fill masking with words exclusion.

    Attributes:
        exclude_words (list[str], optional): Words to exclude from masking.
        mlm_probability (float): Probability of masking tokens during MLM.
        data_collator (CustomWholeWordMaskingDataCollator): Data collator with word exclusion support.

    Methods:
        _tokenize(examples):
            Tokenizes the input examples using the tokenizer.

        _load_model():
            Loads the appropriate model based on whether MLM or causal LM is used.
    """

    def __init__(self, config: General_Special_FillMaskConfig) -> None:
        super().__init__(config)
        self.exclude_words = config.exclude_words
        self.mlm_probability = config.mlm_probability
        self.data_collator = CustomWholeWordMaskingDataCollator(self.tokenizer, self.mlm_probability, self.exclude_words)
    
    def _tokenize(self, examples):
        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        else :
            print("Tokenizer is not fast, so we can't get word_ids mapping to mask the right words")
        return result

    def _load_model(self):
        return AutoModelForMaskedLM.from_pretrained(self.config.model_name)



class CustomWholeWordMaskingDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for whole word masking with word exclusion support.

    Attributes:
        words_not_to_mask (set): A set of words that should not be masked during training.
        
    Methods:
        __call__(features):
            Applies whole word masking to the input features, with specific words excluded.
    """
    def __init__(self, tokenizer, mlm_probability, words_not_to_mask):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.words_not_to_mask = set(words_not_to_mask) if words_not_to_mask is not None else set()

    def __call__(self, features):
        """
        Applies whole word masking to the input features, with specified words excluded from masking.

        Args:
            features (list[dict]): A list of dictionaries representing the input features for training.

        Returns:
            list[dict]: A list of dictionaries with masked input and labels for MLM or causal LM.
        """
        for feature in features:
            word_ids = feature.pop("word_ids")
            input_ids = feature["input_ids"]
            words = self.tokenizer.convert_ids_to_tokens(input_ids)

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, (word_id, word) in enumerate(zip(word_ids, words)):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append((idx, word))

            # Randomly mask words
            mask = np.random.binomial(1, self.mlm_probability, (len(mapping),))
            labels = feature["input_ids"].copy()
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                word_to_mask = self.tokenizer.decode([input_ids[idx] for idx, _ in mapping[word_id]])
                
                # Check if the word should be masked
                if word_to_mask not in self.words_not_to_mask:
                    for idx, _ in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = self.tokenizer.mask_token_id

            feature["labels"] = new_labels

        # Use the default_data_collator to handle the rest
        return default_data_collator(features)