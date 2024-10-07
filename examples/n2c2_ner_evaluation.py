
import os
import sys
import string
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import nlp_attacks
from nlp_attacks.evaluation.evaluation import MLMEvaluation, MLMEvaluationConfig, MLMEvaluationRunParameters

if __name__ == '__main__':

    # ONLY 3 LINES TO MODIFY IN ORDER TO CHANGE THE DATASET USED
    #==========================================================
    data_directory: str = "../data/n2c2_2006"
    ds_loader = nlp_attacks.preprocessing.N2c2NERLoader(data_directory)
    output_folder: str = f"./results/n2c2_ner_fillmask"
    #==========================================================

    dataset = ds_loader.load_dataset()

    #only useful if the dataset has a label column
    dataset = dataset.remove_columns(["label"])

    #Config run
    tokenizer = "bert-base-cased"
    context_length = 512
    
    epochs = [4,8,16,32,64]

    tab_modes = ["indirect_privacy", "direct_privacy", "aw", "all_privacy", "all"]
    model_names = ["anonymous", "direct", "indirect", "all", "baseline", "untrained"]

    for mode in tab_modes:
        
        if mode == "baseline":
            words_not_to_mask = set()
        else:
            words_not_to_mask = ds_loader.get_words_not_to_mask(mode)

        for epoch in epochs:
            output_dir: str = f"{output_folder}/{epoch}_epochs/evaluations"
            model_paths = []
            for t in model_names:
                if(t == "untrained"):
                    model_paths.append(tokenizer)
                else:
                    model_paths.append(f"{output_folder}/{epoch}_epochs/models/bert-base-cased_{t}0")

            if(os.path.exists(f"{output_dir}/{tokenizer}_{mode}_results.json")):
                print(f"{output_dir}/{tokenizer}_{mode}_results.json already exists, skipping...")
                continue


            config = MLMEvaluationConfig(
                model_names=model_names,
                model_paths=model_paths,
                tokenizer_path=tokenizer,
                output_dir=output_dir,
                mode=mode,
                words_not_to_mask=words_not_to_mask,
                context_length=context_length
            )

            evaluator = MLMEvaluation(dataset, config)
            results = evaluator.run(MLMEvaluationRunParameters())