# Made by helain zimmermann following what has been done previously by DABADIE Hugo, MAGNANA Lucas and BERTHELIER Gaspard

from typing import List
import pandas as pd
import numpy as np
import random
import warnings
import torch
import xml.etree.ElementTree as ET

from datasets import Dataset
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
    AutoModelForMaskedLM,
    AutoModelForCausalLM
)


from .Finetuner import Finetuner, FinetunerConfig

from ..utils import anonymize_texts, group_texts

@dataclass
class General_Special_FillMaskConfig(FinetunerConfig):
    """
    Configuration class for the General_Special_FillMask fine-tuning process.

    Attributes:
        exclude_words (list[str], optional): List of words to exclude from masking.
        mlm_probability (float): Probability of masking tokens for MLM.
        tokenizer_name (str): The name of the tokenizer to use.
    """
    mlm: bool = True
    exclude_words: List[str] = None
    mlm_probability: float = 0.15
    tokenizer_name: str = "bert-base-cased"


class General_Special_FillMask(Finetuner):
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
        self.config = config
        self.exclude_words = config.exclude_words
        self.mlm_probability = config.mlm_probability
        if(not self.config.mlm):
            self.custom_token = self.tokenizer.pad_token
            self.gradient_accumulation_steps = 1
            self.warmup_steps = 200
        else:
            self.data_collator = CustomWholeWordMaskingDataCollator(self.tokenizer, self.mlm_probability, self.exclude_words, self.config.mlm)


    
    def _tokenize(self, examples):
        if(self.exclude_words != None and not self.config.mlm):
            modified_texts = anonymize_texts(examples["text"], self.config.exclude_words, anon_str=self.custom_token)
            examples["text"] = [m_t["anonymized_text"] for m_t in modified_texts]       

        result = self.tokenizer(examples["text"])
        if self.config.mlm and self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

        return result

    def _preprocess_dataset(self, examples: Dataset) -> Dataset:
        return group_texts(examples, self.config.context_length)


    def _load_model(self):
        if(self.config.mlm):
            model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # model.resize_token_embeddings(len(self.tokenizer))
        return model



class CustomWholeWordMaskingDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for whole word masking with word exclusion support.

    Attributes:
        words_not_to_mask (set): A set of words that should not be masked during training.
        
    Methods:
        __call__(features):
            Applies whole word masking to the input features, with specific words excluded.
    """
    def __init__(self, tokenizer, mlm_probability, words_not_to_mask, mlm):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.words_not_to_mask = set(words_not_to_mask) if words_not_to_mask is not None else set()
        self.mlm = mlm

        if(not self.mlm):
            self.special_token_ids = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0]),
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id
            ]




    def __call__(self, features):
        """
        Applies whole word masking to the input features, with specified words excluded from masking.

        Args:
            features (list[dict]): A list of dictionaries representing the input features for training.

        Returns:
            list[dict]: A list of dictionaries with masked input and labels for MLM or causal LM.
        """
        vocab_size = len(self.tokenizer.get_vocab())
        for feature in features:
            word_ids = feature.pop("word_ids")
            input_ids = feature["input_ids"]
            words = self.tokenizer.convert_ids_to_tokens(input_ids)
            labels = feature["input_ids"].copy()
            new_labels = [-100] * len(labels)

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
            if(self.mlm):
                mask = np.random.binomial(1, self.mlm_probability, (len(mapping),))
                for word_id in np.where(mask)[0]:
                    word_id = word_id.item()
                    word_to_mask = self.tokenizer.decode([input_ids[idx] for idx, _ in mapping[word_id]])
                    
                    # Check if the word should be masked
                    if word_to_mask not in self.words_not_to_mask:
                        for idx, _ in mapping[word_id]:
                            new_labels[idx] = labels[idx]
                            input_ids[idx] = self.tokenizer.mask_token_id
                feature["labels"] = new_labels
                # print(feature["input_ids"])
                # assert all(0 <= x and x < vocab_size for x in feature["input_ids"])

            #///NOT USED FOR NOW\\\
            else:
                # Masque pour ignorer les tokens spéciaux
                tens_labels = torch.tensor(labels)
                special_tokens_mask = torch.zeros(tens_labels.size(), dtype=torch.bool, device=tens_labels.device)
                for token_id in self.special_token_ids:
                    special_tokens_mask = special_tokens_mask | tens_labels.eq(token_id)

                # Inverse le masque pour avoir True là où les tokens ne sont pas spéciaux
                feature["attention_mask"] = (~special_tokens_mask).tolist()
                feature["labels"] = labels

        # Use the default_data_collator to handle the rest
        return default_data_collator(features)
