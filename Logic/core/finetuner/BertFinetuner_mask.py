import json

import numpy as np
from torch.utils.data import random_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.cluster_cnt = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.raw_dataset = None
        self.dataset = []
        self.label2id = None
        self.id2label = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path) as f:
            self.raw_dataset = json.load(f)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        genres_cnt = {}
        for movie in self.raw_dataset:
            for genre in movie['genres']:
                if genre not in genres_cnt.keys():
                    genres_cnt[genre] = 0
                genres_cnt[genre] += 1
        top_genres = [x[0] for x in sorted(genres_cnt.items(), key=lambda x: x[1], reverse=True)[:self.cluster_cnt]]
        for movie in self.raw_dataset:
            if movie['first_page_summary'] is None:
                continue
            for genre in movie['genres']:
                if genre in top_genres:
                    self.dataset.append(movie)
                    break

        self.label2id = {genre: idx for idx, genre in enumerate(top_genres)}
        self.id2label = {idx: genre for genre, idx in self.label2id.items()}

    def split_dataset(self, test_size=0.1, val_size=0.1):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the test to include in the validation split.
        """
        genres = [movie['genres'] for movie in self.dataset]
        labels = []
        for genre_list in genres:
            label = [0] * self.cluster_cnt
            for genre in genre_list:
                if genre in self.label2id:
                    label[self.label2id[genre]] = 1
            labels.append(label)

        summaries = [movie['first_page_summary'] for movie in self.dataset]
        encodings = self.tokenizer(summaries, padding='max_length', truncation=True, max_length=64, return_tensors='pt')

        dataset = self.create_dataset(encodings, labels)
        proportions = [1 - (test_size + val_size), val_size, test_size]
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        train_data, val_data, test_data = random_split(dataset, proportions)
        return train_data, val_data, test_data

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, train_data, val_data, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              problem_type="multi_label_classification",
                                                              num_labels=self.cluster_cnt,
                                                              id2label=self.id2label,
                                                              label2id=self.label2id)

        args = TrainingArguments(
            "bert-finetuned-sem_eval-english",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            use_cpu=False
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=self.compute_metrics,
        )

        self.trainer = trainer

        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        preds = pred.predictions[0] if isinstance(pred.predictions,
                                                  tuple) else pred.predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(preds))
        y_pred = [1 if p >= 0.3 else 0 for p in probs]
        y_true = pred.labels
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        roc_auc = roc_auc_score(y_true=y_true, y_pred=y_pred, average='micro')
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        return {'F1 Macro-Average Score': f1_macro,
                'ROC Area Under Curve': roc_auc,
                'Accuracy': accuracy}

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        return self.trainer.evaluate()

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        model = self.trainer.model
        model.save_pretrained(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """

        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
