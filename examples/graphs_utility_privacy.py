import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


tab_datasets = ["aphp_pseudo_fillmask"]

# Utility and privacy metrics for the experiments

graph_dir = Path("graphs")
graph_dir.mkdir(parents=True, exist_ok=True)


dict_results = {}

epochs = [4,8,16,32,64]

for dataset in tab_datasets:
    graph_dir_dataset = Path(f"{graph_dir.name}/{dataset}")
    graph_dir_dataset.mkdir(parents=True, exist_ok=True)
    dict_results[dataset] = {"Direct_privacy": [], "Indirect_privacy": [], "all_Privacy": [], "all": [], "aw": []}
    for epoch in epochs:
        for key in dict_results[dataset]:
            with open(f"results/{dataset}/{epoch}_epochs/evaluations/camembert-base_{key.lower()}_results.json", 'r') as file:
                dict_results[dataset][key].append(json.load(file))


# Convert utility and privacy to lists for each model

for dataset in dict_results:
    utilities = {}
    for utility in ["all", "aw"]:
        utilities[utility] = {}
        for model in dict_results[dataset][utility][0]["accuracy"]:
            utilities[utility][model] = [dict_results[dataset][utility][i]["accuracy"][model] for i in range(len(dict_results[dataset][utility]))]

    for privacy in dict_results[dataset]:
        if("privacy" not in privacy.lower()):
            continue

        privacies = {}
        for model in dict_results[dataset][privacy][0]["accuracy"]:
            privacies[model] = [1 - dict_results[dataset][privacy][i]["accuracy"][model] for i in range(len(dict_results[dataset][privacy]))]



        for utility in utilities:
            plt.figure(figsize=(12, 8))
            # Plot points for each model and add epoch numbers inside different colored shapes
            for i, epoch in enumerate(epochs):
                plt.scatter(utilities[utility]["all"][i], privacies["all"][i], color='purple', s=400, marker='d')
                plt.text(utilities[utility]["all"][i], privacies["all"][i], str(epoch), color='white', ha='center', va='center', fontsize=10, fontweight='bold')

                plt.scatter(utilities[utility]["direct"][i], privacies["direct"][i], color='blue', s=400, marker='o')
                plt.text(utilities[utility]["direct"][i], privacies["direct"][i], str(epoch), color='white', ha='center', va='center', fontsize=10, fontweight='bold')

                plt.scatter(utilities[utility]["indirect"][i], privacies["indirect"][i], color='black', s=400, marker='h')
                plt.text(utilities[utility]["indirect"][i], privacies["indirect"][i], str(epoch), color='white', ha='center', va='center', fontsize=10, fontweight='bold')
                
                plt.scatter(utilities[utility]["baseline"][i], privacies["baseline"][i], color='green', s=400, marker='s')
                plt.text(utilities[utility]["baseline"][i], privacies["baseline"][i], str(epoch), color='white', ha='center', va='center', fontsize=10, fontweight='bold')
                
                plt.scatter(utilities[utility]["anonymous"][i], privacies["anonymous"][i], color='red', s=400, marker='^')
                plt.text(utilities[utility]["anonymous"][i], privacies["anonymous"][i], str(epoch), color='white', ha='center', va='center', fontsize=10, fontweight='bold')


            # Adding labels and title
            plt.title(dataset.split("_")[0], fontsize=24)
            xlab = 'Fillmask Utility'
            if(utility == "all"):
                xlab += " (no identifiers)"
            elif("aw" in utility):
                xlab += " (all words)"
            plt.xlabel(xlab, fontsize=24, labelpad=16)

            plt.ylabel(f'{privacy.replace("_", " ").replace("all", "")} (Greater is better)', fontsize=24, labelpad=20)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            # adjust legend position , fontsize and size
            plt.legend(['PPMLM-BERT', 'DPPMLM-BERT', 'IPPMLM-BERT', 'MLM-BERT', 'MLMA-BERT'], loc='lower left', fontsize=16)
            plt.grid(True, which="both", ls="-.", linewidth=0.5, color='lightgray')

            # Save the plot
            plt.savefig(f'{graph_dir}/{dataset}/{"all_words_" if "aw" in utility else ""}{privacy.lower()}.png')

for dataset in dict_results:
    for key in dict_results[dataset]:
        if("privacy" not in key.lower()):
            continue

        # Plotting privacy accuracy over epochs (with log scale on x-axis)
        plt.figure(figsize=(12, 8))

        # Plot lines for each model
        plt.plot(epochs, [1 - data["accuracy"]["all"] for data in dict_results[dataset][key]], color='purple', linestyle=(0, (3, 1, 1, 1, 1, 1)))
        plt.plot(epochs, [1 - data["accuracy"]["direct"] for data in dict_results[dataset][key]], color='blue', linestyle='dashed')
        plt.plot(epochs, [1 - data["accuracy"]["indirect"] for data in dict_results[dataset][key]], color='black', linestyle=(0, (3, 5, 1, 5, 1, 5)))
        plt.plot(epochs, [1 - data["accuracy"]["baseline"] for data in dict_results[dataset][key]], color='green', linestyle='dotted')
        plt.plot(epochs, [1 - data["accuracy"]["anonymous"] for data in dict_results[dataset][key]], color='red', linestyle='dashdot')
        plt.plot(epochs, [1 - data["accuracy"]["untrained"] for data in dict_results[dataset][key]], color='orange', linestyle='solid')

        # Adding labels and title
        plt.title(dataset.split("_")[0], fontsize=24)
        plt.xlabel('Epochs', fontsize=24, labelpad=16)
        plt.ylabel(f'{key.replace("_", " ").replace("all", "")} (Greater is better)', fontsize=24, labelpad=20)
        plt.legend(['PPMLM-BERT', 'DPPMLM-BERT', 'IPPMLM-BERT', 'MLM-BERT', 'MLMA-BERT', 'BERT'], fontsize=16)

        # Define specific x-axis ticks
        plt.xticks([4, 8, 16, 32, 64], [4, 8, 16, 32, 64], fontsize=16)
        plt.yticks(fontsize=16)

        # Add grid for better readability
        plt.grid(True, which="both", ls="-.", linewidth=0.5, color='lightgray')

        # Save the plot
        plt.savefig(f'{graph_dir}/{dataset}/{key.lower()}_epochs.png')