from typing import List, Tuple
from tqdm import tqdm
import string
import re

def get_words_to_persons_dict(texts: List[str], text_to_person: List[int]) -> dict:
    """
    Compute a dictionary that links each word in a list of texts to the persons it can indirectly identify.

    The dictionary keys are words from the texts, and the values are lists of person IDs 
    (from `text_to_person`) representing the people that the words can identify indirectly. 
    For example, if a word appears in texts associated with three different persons, the word 
    will have a list of three person IDs as its value.

    Args:
        texts: The texts to analyze.
        text_to_person (list[int]): A list of integers where each integer represents the person 
                                    identifiable by the corresponding text in the dataset.

    Returns:
        dict: A dictionary mapping words to lists of person IDs they may indirectly identify.
    """

    words_to_persons = {}

    for i in tqdm(range(len(texts)), desc="Words to person"):
        t = texts[i]
        for punc in string.punctuation:
            t = t.replace(punc, f" {punc} ")
        t = t.replace("  ", " ")
        words = t.split()
        for word in words:
            lc_word = word
            if lc_word not in words_to_persons:
                words_to_persons[lc_word] = [text_to_person[i]]
            elif text_to_person[i] not in words_to_persons[lc_word]:
                words_to_persons[lc_word].append(text_to_person[i])

    return words_to_persons


def anonymize_texts(texts: List[str], identifiers: set) -> List[dict]:
    """
    Anonymize a list of texts by replacing all words that match the provided identifiers with 'X'.

    This function goes through each text and replaces words that are found in the `identifiers` set with 'X'.
    The anonymized texts are then returned in a list of dictionaries.

    Args:
        texts (list[str]): A list of texts to be anonymized.
        identifiers (set): A set of words that need to be anonymized in the texts.

    Returns:
        list[dict]: A list of dictionaries with the anonymized text under the key 'anonymized_text'.
    """
    anonymized_texts = []
    an_number = 0
    for t in texts:
        for punc in string.punctuation:
            t = t.replace(punc, f" {punc} ")
        t = t.replace("  ", " ")
        anonymized_t = []
        words = t.split()
        for w in words:
            if(w in identifiers):
                anonymized_t.append("X")
                an_number += 1
            else:
                anonymized_t.append(w)
        anonymized_texts.append({"anonymized_text": " ".join(anonymized_t)})

    return anonymized_texts


def group_texts(examples, chunk_size):
    # Concaténation de tous les textes
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Calcule la longueur des textes concaténés
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Nous laissons tomber le dernier morceau s'il est plus petit que chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Fractionnement par chunk de max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Créer une nouvelle colonne d'étiquettes
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def get_identifiers(texts: List[str], entities: list, mode: str, k: int = 2, 
                    excluded_entities: list = [], text_to_person: list = None, no_punc: bool = True) -> Tuple[set, set]:
    """
    Identify words in the texts that are direct and/or indirect identifiers of individuals based on the provided mode.
    It can also exclude specific entities and optionally remove punctuation from identifiers. 
    The output consists of two sets: one for identifiers and one for non-identifiers.

    Args:
        texts (list[str]): The list of texts to analyze.
        entities (list): A list of named entities associated with the texts.
        mode (str): The mode for identifying words. Options include 'direct', 'indirect', 'all'.
        k (int, optional): The minimum number of individuals associated with a word to not classify it as an indirect identifier. Default is 2.
        excluded_entities (list, optional): Entities to exclude from direct identifier detection. Default is an empty list.
        text_to_person (list, optional): A list mapping each text to a person. If not provided, each text is assumed to be linked to a person.
        no_punc (bool, optional): Whether to remove punctuation from the returned words. Default is True.

    Returns:
        tuple[set, set]: Two sets: one with identifiers and one with non-identifiers.
    """            
    d_not_identifiers = set()
    d_identifiers = set()
    ind_not_identifiers = set()
    ind_identifiers = set()
    all_unique_words = set()

    if(text_to_person == None):
        text_to_person = list(range(len(texts)))

    if("indirect" in mode or "all" in mode):
        words_to_persons = get_words_to_persons_dict(texts, text_to_person)
        for word in words_to_persons:
            if(len(words_to_persons[word]) < k):
                ind_identifiers.add(word)
            else:
                ind_not_identifiers.add(word)
        all_unique_words = set(words_to_persons.keys())
    

    if(("direct" in mode and "indirect" not in mode) or "all" in mode):
        d_not_identifiers = []
        d_identifiers = []
        for text, entities in tqdm(zip(texts, entities), total=len(texts), desc="Direct identifiers"):
            words = text.split()
            for word, entity in zip(words, entities):
                all_unique_words.add(word)
                if entity not in excluded_entities:
                    d_identifiers.append(word)
                else:
                    d_not_identifiers.append(word)

        d_identifiers_dict = {}
        d_not_identifiers_dict = {}

        for word in d_not_identifiers:
            d_not_identifiers_dict[word] = d_not_identifiers_dict.get(word, 0) + 1

        for word in d_identifiers:
            d_identifiers_dict[word] = d_identifiers_dict.get(word, 0) + 1

        intersection = intersection = set(d_not_identifiers_dict.keys()) & set(d_identifiers_dict.keys())

        d_identifiers = set(d_identifiers)
        d_not_identifiers = set(d_not_identifiers)


        for word in intersection:
            if d_not_identifiers_dict[word] > d_identifiers_dict.get(word, 0):
                d_identifiers.remove(word)
            elif d_not_identifiers_dict[word] < d_identifiers_dict.get(word, 0):
                d_not_identifiers.remove(word)

                

    identifiers = ind_identifiers.union(d_identifiers)

    if(no_punc):
        copy_identifiers = identifiers.copy()
        for ident in tqdm(copy_identifiers, desc="Cleaning ids", total=len(copy_identifiers)):
            new_idents = []
            spl = re.split(r"[!\"#$%&'()*+-./:;<=>?@[\]^_`{|}~]", ident)
            new_idents.extend(spl)
            identifiers.remove(ident)
            identifiers.update(new_idents)

        copy_all_unique_words = all_unique_words.copy()
        for ident in tqdm(copy_all_unique_words, desc="Cleaning all unique words", total=len(copy_all_unique_words)):
            new_idents = []
            spl = re.split(r"[!\"#$%&'()*+-,./:;<=>?@[\]^_`{|}~]", ident)
            new_idents.extend(spl)
            all_unique_words.remove(ident)
            all_unique_words.update(new_idents)

        identifiers -= set(["", "A", "s"]) #some misinterpreted identifiers we found

    not_identifiers = all_unique_words.symmetric_difference(identifiers)
    not_identifiers.update(list(string.punctuation))



    return identifiers, not_identifiers