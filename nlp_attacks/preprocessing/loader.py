import os
from abc import ABC, abstractmethod

import numpy as np
from datasets import Dataset
from ..utils import get_identifiers



class Loader(ABC):
    """
    Abstract base class for loading datasets. It defines common methods and attributes 
    related to loading and processing datasets.

    Attributes:
        directory (str): Path to the directory where dataset files are stored.
        dataset (Dataset): The loaded dataset, initialized as None.
        text_to_person (list): Mapping between texts and corresponding patient IDs (if applicable).
        entities (list): A list of named entities found in the dataset, needed to compute the direct identifiers in the
            get_words_not_to_mask function.
        excluded_entities (list): A list of entity types to be excluded during the processing of direct identifiers.
    """
    def __init__(self, directory: str) -> None:
        """
        Initializes the Loader object with the specified directory.

        Args:
            directory (str): The path to the directory containing the dataset files.
        """
    
        self.directory = directory

        self.dataset = None
        self.text_to_person = None
        self.entities = []
        self.excluded_entities = []


    def get_words_not_to_mask(self, mode: str, k: int = 2, no_punc: bool = True) -> set:
        """
        Retrieves the set of words that should not be masked during training, 
        based on the provided mode.

        This method loads the dataset (if not already loaded) and identifies which words
        should not be masked based on identifiers (directs and/or indirects depending on the 
        specified mode)

        Args:
            mode (str): The mode to determine whether to focus on identifiers and/or non-identifiers.
                        Possible values include "direct", "indirect" or "all". Adding "privacy" return the non identifiers words.
            k (int, optional): Minimum 
            no_punc (bool, optional): Whether to exclude punctuation from the identifiers. Default is True.

        Returns:
            set: A set of words that should not be masked during training.
        """
        
        if(self.dataset == None):
            self.load_dataset()

        identifiers, not_identifiers = get_identifiers(self.dataset["text"], self.entities, mode,
            k=k, excluded_entities=self.excluded_entities, text_to_person=self.text_to_person, no_punc=no_punc)

            
        if("privacy" in mode):
            return not_identifiers
        else:
            return identifiers


    @abstractmethod
    def load_dataset(self) -> Dataset:
        """
        Abstract method that must be implemented by subclasses to load a specific dataset.

        This method should return a Hugging Face `Dataset` object with a text column at minimum. Each subclass 
        needs to define its own version of this method based on how the dataset is 
        structured and loaded.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Must be implemented in subclasses")
