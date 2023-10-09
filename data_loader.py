from datasets import load_dataset
from transformers import AutoTokenizer

class banking:
    def __init__(self, task, tokenizer):
        self.task = task
        self.tokenizer = tokenizer
        # Dataset id from huggingface.co/dataset
        self.dataset_id = "banking77"
        #
        self.raw_dataset = self.get_dataset()
        #
        self.tokenized_dataset, self.labels, self.num_labels, self.label2id, self.id2label = self.transform_id_label()


    def get_dataset(self):
        raw_dataset = load_dataset(self.dataset_id)
        if self.task == 'train':
            raw_dataset = raw_dataset['train']
        else:
            raw_dataset = raw_dataset['test']
        return raw_dataset

    # Tokenize helper function
    def tokenize(self,batch):
        return self.tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")

    def transform_id_label(self):
        raw_dataset = self.raw_dataset.rename_column("label", "labels")  # to match Trainer
        tokenized_dataset = raw_dataset.map(self.tokenize, batched=True, remove_columns=["text"])

        labels = tokenized_dataset.features["labels"].names

        num_labels = len(labels)
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        return tokenized_dataset, labels, num_labels, label2id, id2label

