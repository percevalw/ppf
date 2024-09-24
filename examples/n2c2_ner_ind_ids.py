import os
import sys
import string
from typing import Literal
from datasets import Dataset
import scispacy
import spacy
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import collections
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import nlp_attacks

if __name__ == '__main__':

    '''
    This file is mainly used to quantitatively and qualitatively observe the indirect identifiers. For now it computes
    a CDF of the indirect identifiers, classifies them using a bio NER model and computes the proportion of individuals 
    that can be indirectly reidentified using an organ or a cancer. It also prints the indirect identifiers classified 
    as a cancer to verify the coherence of the classification.
    '''
    
    result_name_dir = "n2c2_ner_fillmask"
    data_directory: str = "../data/n2c2_ner"
    ds_loader = nlp_attacks.preprocessing.N2c2NERLoader(data_directory)

    output_dir = f"results/{result_name_dir}"
    graph_dir = f"graphs/{result_name_dir}"
    os.makedirs(f"{output_dir}/ind_ids/", exist_ok=True)
    os.makedirs(f"{graph_dir}", exist_ok=True)

    
    dataset = ds_loader.load_dataset()
    ind_identifiers = ds_loader.get_words_not_to_mask("indirect")
    d_identifiers = ds_loader.get_words_not_to_mask("direct")



    nlp = spacy.load("en_ner_bionlp13cg_md")
    if(os.path.exists(f"{data_directory}/bio_ner.json")):
        with open(f"{data_directory}/bio_ner.json", "r") as f:
            dict_bio_ner = json.load(f)
    else:
        dict_bio_ner = {}
        for i in tqdm(range(len(dataset)), desc="Computing bio NER"):
            text = dataset["text"][i]
            for punc in (string.punctuation):
                text = text.replace(punc, f" {punc} ")
            text = text.replace("  ", " ")

            dict_bio_ner[i] = {"texts": [], "labels": []}
            doc = nlp(text)
            for ent in doc.ents:
                dict_bio_ner[i]["texts"].append(ent.text)
                dict_bio_ner[i]["labels"].append(ent.label_)

            with open(f"{data_directory}/bio_ner.json", "w") as f:
                json.dump(dict_bio_ner, f)


    num_ids_pers = np.zeros(max(ds_loader.text_to_person)+1)
    organ_id_pers = np.zeros(max(ds_loader.text_to_person)+1)
    cancer_id_pers = np.zeros(max(ds_loader.text_to_person)+1)

    dict_ids_type = {}

    cancer_list = []

    for lab in nlp.get_pipe('ner').labels:
        dict_ids_type[lab] = 0

    for i in range(len(dataset)):
        text = dataset["text"][i]
        num_ids = 0
        for punc in (string.punctuation):
            text = text.replace(punc, f" {punc} ")
        text = text.replace("  ", " ")
        words = text.split()

        for word in words:
            if(word in ind_identifiers):
                num_ids += 1
        num_ids_pers[ds_loader.text_to_person[i]] += num_ids

        unique_identifiers = set()
        for j, ent in enumerate(dict_bio_ner[str(i)]["texts"]):
            label = dict_bio_ner[str(i)]["labels"][j]
            cleant_ent = ent.lower()
            cleant_ent = " ".join(re.split(r"[!\"#$%&'()*+-,./:;<=>?@[\]^_`{|}~]", cleant_ent))
            for ent_part in ent.split():
                if(ent_part != ent_part.upper() and ent_part in ind_identifiers and ent_part not in d_identifiers and cleant_ent not in unique_identifiers):
                    dict_ids_type[label] += 1
                    unique_identifiers.add(cleant_ent)
                    if(label == "ORGAN"):
                        organ_id_pers[ds_loader.text_to_person[i]] = 1
                    elif(label == "CANCER"):
                        cancer_list.append(ent)
                        cancer_id_pers[ds_loader.text_to_person[i]] = 1

    print(cancer_list)

    d = {"Proportion of patients reindentifiable": [cancer_id_pers.mean(), organ_id_pers.mean()], "Type": ["CANCER", "ORGAN"]}
    df = pd.DataFrame(data=d)
    sns.barplot(df, x="Proportion of patients reindentifiable", y="Type")
    plt.xticks(rotation=33)
    plt.savefig(f"{graph_dir}/proportion_reids_patients.png", bbox_inches="tight")
    plt.clf()

    sns.displot(num_ids_pers, kind="ecdf").set(title="Distribution of the number of indirect identifiers",
    xlabel="Number of indirect identifiers", ylabel="Proportion of patients")
    plt.savefig(f"{graph_dir}/cdf_ind_ids.png", bbox_inches="tight")
    plt.clf()


    print(dict_ids_type)
    d = {"Number of unique indirect identifiers": list(dict_ids_type.values()), "Type": list(dict_ids_type.keys())}
    df = pd.DataFrame(data=d)
    plt.figure(figsize=(25, 5))
    sns.barplot(df, x="Type", y="Number of unique indirect identifiers")
    plt.xticks(rotation=33)
    plt.savefig(f"{graph_dir}/type_unique_indirect_ids.png", bbox_inches="tight")
    plt.clf()

