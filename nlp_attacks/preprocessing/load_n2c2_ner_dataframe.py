import pandas as pd
from datasets import Dataset
import xml.etree.ElementTree as ET

from .loader import Loader


class N2c2NERLoader(Loader):
    """
    A class to load and process the n2c2 de-identified named entity recognition (NER) dataset from XML files.

    Attributes:
        directory (str): Path to the directory containing the n2c2 NER file (deid_surrogate_train_all_version2.xml).
        max_data (int): Maximum number of records to load. Default is -1 (load all).
        excluded_entities (list): List of entities to be excluded during the processing of direct identifiers 
            in the get_words_not_to_mask function of the Loader class.
        dataset (Dataset): The processed dataset, initialized as None.
        entities (list): List of named entities found in the dataset, it is needed by the get_words_not_to_mask function.
        text_to_person (list): Mapping between texts and their corresponding patient numbers.
        _patient_names (list): List of extracted patient names to compute the text_to_person list.
    """
    def __init__(self, directory: str, max_data: int = -1, mode_test: bool = False) -> None:
        """
        Initialize the N2c2NERLoader class with the given directory and max_data.

        Args:
            directory (str): The directory containing the XML files.
            max_data (int, optional): The maximum number of records to load. Defaults to -1 (load all records).
        """
        super().__init__(directory)
        self.max_data = max_data
        if(mode_test):
            self.tab_files = ["deid_surrogate_test_all_groundtruth_version2.xml"]
        else:
            self.tab_files = ["deid_surrogate_train_all_version2.xml"]

        self.excluded_entities = ["O"]
        self._patient_names = []

    def load_dataset(self) -> Dataset:
        """
        Load the n2c2 dataset and convert it to a Hugging Face Dataset object.

        This method calls the `_load_raw_data` function to retrieve the raw data and then 
        converts it to a Hugging Face Dataset format. The dataset is cached in the class attribute.

        Returns:
            Dataset: A Hugging Face Dataset object containing text and label columns.
        """
        if(self.dataset == None):
            raw_data = self._load_raw_data()
            self.dataset = Dataset.from_dict({"text": raw_data["text"].values, "label": raw_data["label"].values})
            self.entities = raw_data["named_entities"].values

        return self.dataset


    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw text, labels, and named entities from XML files in the directory.

        This method reads XML files and extracts the texts, labels, and named entities.
        The number of text files processed is limited by the `max_data` parameter.

        Returns:
            pd.DataFrame: A DataFrame with columns for text, label, and named entities.

        Raises:
            ValueError: If the number of texts does not match the number of labels.
        """
        texts = []
        labels = []
        named_entities = []
        possible_labels = ['ADRESSE',
         'DATE',
         'DATE_NAISSANCE',
         'HOPITAL',
         'IPP',
         'MAIL',
         'NDA',
         'NOM',
         'PRENOM',
         'SECU',
         'TEL',
         'VILLE',
         'ZIP']
        number_of_text_files = 0
        for text_file in self.tab_files:
            if text_file.endswith(".xml"):
                number_of_text_files += 1
                text_file_path = f"{self.directory}/{text_file}"
                texts, labels, named_entities = self._get_text_and_labels(text_file_path, possible_labels)

        if len(texts) != len(labels):
            raise ValueError(f"There are not the same number of texts ({len(texts)})"
                                    f"as the number of labels ({len(labels)})")

        return pd.DataFrame({"text": texts, "label": labels, "named_entities": named_entities})


    def _get_text_and_labels(self, file_path: str, possible_labels: list) -> tuple:
        """
        Parse a single XML file to extract text and NER labels for each record.

        This method reads through each <RECORD> element in the XML file, extracting the 
        text and its corresponding NER labels (such as "PATIENT", "DOCTOR", etc.). It also handles 
        the processing of the text_to_person attribute by checking the first iteration of the name
        of the patient for each <RECORD>. Finally, each label is affiliated with a named in the
        named_entities list.

        Args:
            file_path (str): The path to the XML file.
            possible_labels (list): A list of possible label types to identify in the text.

        Returns:
            tuple: A tuple containing three lists: texts, labels, and named entities.

        Raises:
            ValueError: If there is a mismatch between the number of words and the number of labels in a record.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        texts = []
        labels = []
        named_entities = []
        num_record = 0
        self.text_to_person = []
        for record_idx, record in enumerate(root.findall(".//RECORD")):
            if num_record == self.max_data:
                break
            num_record += 1
            patient_num_extracted = False
            text = []
            label = []
            named_entity = []
            has_patient_id = False
            for t in record.findall("TEXT"):
                for phi in t.findall("PHI"):
                    if(phi.attrib["TYPE"] == "PATIENT" and not patient_num_extracted):
                        has_patient_id = True
                        if(phi.text not in self._patient_names):
                            self._patient_names.append(phi.text)
                        self.text_to_person.append(self._patient_names.index(phi.text))
                        patient_num_extracted = True

                    entities = phi.text.split()
                    l = possible_labels.index(phi.attrib["TYPE"])
                    for i in range(len(entities)):
                        text.append(entities[i])
                        named_entity.append(phi.attrib["TYPE"])
                        if(i==0):
                            label.append(str(l+1))
                        else:
                            label.append(str(l+2))
                    text.extend(phi.tail.split())
                    label.extend(["0"]*len(phi.tail.split()))
                    named_entity.extend(["O"]*len(phi.tail.split()))
            if not has_patient_id:
                self._patient_names.append(record_idx)
                self.text_to_person.append(record_idx)
            if(len(text) == len(label)):
                texts.append(" ".join(text))
                labels.append(" ".join(label))
                named_entities.append(named_entity)
                if(num_record > len(self.text_to_person)):
                    self.text_to_person.append(max(self.text_to_person)+1)
                    self._patient_names.append("ano")
            else:
                raise ValueError(f"There are not the same number of label ({len(label)})"
                                 f"as the number of words ({len(text)}) in the record {record.attrib['ID']}")
        return texts, labels, named_entities
