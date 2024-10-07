import os
import sys
import string
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import Dataset
import nlp_attacks

if __name__ == '__main__':

    # ONLY 3 LINES TO MODIFY IN ORDER TO CHANGE THE DATASET USED
    #==========================================================
    data_directory: str = "../data/n2c2_2006"
    ds_loader = nlp_attacks.preprocessing.N2c2NERLoader(data_directory)
    output_folder: str = f"./results/n2c2_ner_fillmask"
    #==========================================================


    # Run parameters
    test_size: float = 0

    #Config run
    seed: int = 42
    context_length = 512
    model_name = "bert-base-cased"
    
    epochs_runs = [4,8,16,32,64]

    '''
    These modes are used in order to decide which words will not be masked
    and to name the saved models to differentiate them.
    '''
    tab_modes = ["anonymous", "direct", "indirect", "all", "baseline"]


    for mode in tab_modes:

        dataset = ds_loader.load_dataset()

        #only useful if the dataset has a label column
        dataset = dataset.remove_columns(["label"])

        if mode == "anonymous":
            words_not_to_mask = None
            anonymized_texts = nlp_attacks.utils.anonymize_texts(dataset["text"], ds_loader.get_words_not_to_mask(mode="all"))
            dataset = Dataset.from_list(anonymized_texts)
            dataset = dataset.rename_column("anonymized_text", "text")

        elif mode == "baseline":
            words_not_to_mask = None
        else:
            words_not_to_mask = ds_loader.get_words_not_to_mask(mode)


        for epochs in epochs_runs:

            # Config fill mask
            output_dir = f"{output_folder}/{epochs}_epochs/models"                        
            ouput_name = f"{mode}"


            f_config = nlp_attacks.finetuners.General_Special_FillMaskConfig(model_name, seed, context_length, [], 1e-4, exclude_words=words_not_to_mask, tokenizer_name=model_name)
            finetuner = nlp_attacks.finetuners.General_Special_FillMask(f_config)

            models, metrics = finetuner.run(dataset, test_size, epochs, Path(output_dir), output_name=ouput_name)
            print(metrics)