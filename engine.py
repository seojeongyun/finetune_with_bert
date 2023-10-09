import evaluate
import numpy as np
import torch

from transformers import AutoTokenizer
from data_loader import banking
from transformers import AutoModelForSequenceClassification
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments


class engine:
    def __init__(self):
        self.device = torch.device("cpu")
        # Model id to load the tokenizer
        self.model_id = "bert-base-uncased"
        # Load Tokenizer
        self.tokenizer = self.get_tokenizer()
        #
        self.train_dataload = self.get_loader('train', self.tokenizer)
        self.test_dataload = self.get_loader('test', self.tokenizer)
        #
        self.model = self.get_model()
        #
        self.metric = evaluate.load("f1")
        #
        self.repo_id, self.train_args, self.trainer = self.get_args()

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return tokenizer

    def get_loader(self, task, tokenizer):
        if task == 'train':
            loader = banking(task, tokenizer)
        else:
            loader = banking(task, tokenizer)

        return loader

    def get_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
                                                                   num_labels=self.train_dataload.num_labels,
                                                                   label2id=self.train_dataload.label2id,
                                                                   id2label=self.train_dataload.id2label)
        return model

        # Metric helper method

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels, average="weighted")

    def get_args(self):
        # Id for remote repository
        repository_id = "bert-base-banking77-pt2-jy"

        # Define training args
        training_args = TrainingArguments(
            output_dir=repository_id,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=3,
            # PyTorch 2.0 specifics
            fp16=False,  # bfloat16 training
            torch_compile=True,  # optimizations
            optim="adamw_torch_fused",  # improved optimizer
            # logging & evaluation strategies
            logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=200,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            # push to hub parameters
            report_to="tensorboard",
            push_to_hub=True,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),

        )

        # Create a Trainer instance
        trainer = Trainer(
            model=self.model.to(self.device),
            args=training_args,
            train_dataset=self.train_dataload.tokenized_dataset,
            eval_dataset=self.test_dataload.tokenized_dataset,
            compute_metrics=self.compute_metrics,
        )

        return repository_id, training_args, trainer
