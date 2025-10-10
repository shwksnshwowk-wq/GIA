import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup,
    BertForSequenceClassification
)
from datasets import DatasetDict
import pandas as pd
import os

class DPFederatedBERT:
    """BERT federated learning client with differential privacy protection"""
    
    def __init__(self, model, epsilon=1.0, delta=1e-5, max_grad_norm=1.0, target_sensitivity=1.0):
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.target_sensitivity = target_sensitivity
        self.sigma = self._calculate_noise_scale()
        
    def _calculate_noise_scale(self):
        return self.target_sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=self.max_grad_norm,
            norm_type=2
        )
    
    def add_dp_noise_to_gradients(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.sigma,
                        size=param.grad.shape,
                        device=param.grad.device
                    )
                    param.grad += noise
    
    def local_training_with_dp(self, dataloader, optimizer, scheduler, device, accumulation_steps=4):
        self.model.train()
        self.model.to(device)
        
        total_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                self.clip_gradients()
                self.add_dp_noise_to_gradients()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
        if len(dataloader) % accumulation_steps != 0:
            self.clip_gradients()
            self.add_dp_noise_to_gradients()
            optimizer.step()
            scheduler.step()
            
        return total_loss / len(dataloader)

class BERTFederatedLearningServer:
    def __init__(self, global_model, client_models):
        self.global_model = global_model
        self.client_models = client_models
        
    def aggregate_models(self):
        global_state = self.global_model.state_dict()
        
        for key in global_state.keys():
            if 'weight' in key or 'bias' in key:
                client_params = []
                for client_model in self.client_models:
                    client_state = client_model.state_dict()
                    if key in client_state:
                        client_params.append(client_state[key])
                
                if client_params:
                    global_state[key] = torch.stack(client_params).mean(dim=0)
        
        self.global_model.load_state_dict(global_state)
        for client_model in self.client_models:
            client_model.load_state_dict(global_state)

def load_local_cola_dataset(data_dir):
    """load the local cola"""
    # check the save path
    possible_paths = [
        os.path.join(data_dir, 'train.tsv'),
        os.path.join(data_dir, 'in_domain_train.tsv'),
        os.path.join(data_dir, 'raw/in_domain_train.tsv')
    ]
    
    train_path = None
    valid_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            train_path = path
            valid_path = path.replace('train', 'dev').replace('train', 'valid')
            if not os.path.exists(valid_path):
                valid_path = path.replace('in_domain_train', 'in_domain_dev')
            break
    
    if train_path is None or not os.path.exists(valid_path):
        print(f"The CoLA dataset file cannot be found. Please place the train.tsv and dev.tsv files in the {data_dir} directory.")
        return None
    
    print(f"find train dataset: {train_path}")
    print(f"find test dataset: {valid_path}")
    
    # load data
    train_df = pd.read_csv(train_path, sep='\t', header=0, quoting=3)
    valid_df = pd.read_csv(valid_path, sep='\t', header=0, quoting=3)
    
    print(f"size of train data: {len(train_df)}")
    print(f"size of valid data: {len(valid_df)}")
    print("name of train data columns:", train_df.columns.tolist())
    
    # standard the columns
    column_mapping = {}
    for col in train_df.columns:
        if 'sentence' in col.lower() or 'text' in col.lower():
            column_mapping[col] = 'sentence'
        elif 'label' in col.lower() or 'acceptability' in col.lower():
            column_mapping[col] = 'labels'
    
    train_df = train_df.rename(columns=column_mapping)
    valid_df = valid_df.rename(columns=column_mapping)
    
    # Ensure that the necessary columns are present
    if 'sentence' not in train_df.columns:
        # Try using the third column as the sentence (in the CoLA standard format)
        if len(train_df.columns) >= 4:
            train_df = train_df.rename(columns={train_df.columns[3]: 'sentence'})
            valid_df = valid_df.rename(columns={valid_df.columns[3]: 'sentence'})
    
    if 'labels' not in train_df.columns:
        # Try to use the first column as the label (in the CoLA standard format)
        if len(train_df.columns) >= 2:
            train_df = train_df.rename(columns={train_df.columns[1]: 'labels'})
            valid_df = valid_df.rename(columns={valid_df.columns[1]: 'labels'})
    
    from datasets import Dataset
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(valid_df)
    })
    
    return dataset

def prepare_cola_federated_data(tokenizer, dataset, num_clients=3, max_length=128):
    """Prepare the CoLA data for FL"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    # tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Split the dataset
    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["validation"]
    
    client_datasets = []
    total_samples = len(train_dataset)
    samples_per_client = total_samples // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else total_samples
        client_data = train_dataset.select(range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    print(f"CoLA Split: {num_clients} clients with {samples_per_client} samples")
    
    return client_datasets, validation_dataset

def create_dataloaders(client_datasets, validation_dataset, batch_size=8):
    """create dataloaders"""
    client_dataloaders = []
    for dataset in client_datasets:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(dataloader)
    
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    return client_dataloaders, validation_dataloader

def evaluate_model(model, dataloader, device):
    """evaluate the performance of model"""
    model.eval()
    model.to(device)
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'sentence'}
            outputs = model(
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                labels=batch.get('labels')
            )
            
            loss = criterion(outputs.logits, batch["labels"])
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)
    
    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def main():
    # para setting
    num_clients = 2
    batch_size = 8
    learning_rate = 2e-5
    num_epochs_local = 2
    num_rounds = 3
    accumulation_steps = 2
    
    # DP-para
    epsilon = 10.0
    delta = 1e-5
    max_grad_norm = 1.0
    
    # init BERT
    model_name = "/home/csluo/FL-LLM/models/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # load local CoLA
    data_dir = "./cola_public"
    dataset = load_local_cola_dataset(data_dir)
    
    if dataset is None:
        print("can not load the dataset, pls check the path")
        return
    
    # init FL data
    client_datasets, validation_dataset = prepare_cola_federated_data(
        tokenizer, dataset, num_clients
    )
    client_dataloaders, validation_dataloader = create_dataloaders(
        client_datasets, validation_dataset, batch_size
    )
    
    # init model
    global_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    client_models = []
    for _ in range(num_clients):
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        client_models.append(model)
    
    # init server and DP clients
    server = BERTFederatedLearningServer(global_model, client_models)
    dp_clients = []
    for i in range(num_clients):
        dp_client = DPFederatedBERT(
            model=client_models[i],
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm
        )
        dp_clients.append(dp_client)
    
    # cycle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    for round_idx in range(num_rounds):
        print(f"\n--- FL Round {round_idx + 1}/{num_rounds} ---")
        
        client_losses = []
        for client_idx, dp_client in enumerate(dp_clients):
            print(f"Client {client_idx + 1} training...")
            
            dataloader = client_dataloaders[client_idx]
            optimizer = AdamW(dp_client.model.parameters(), lr=learning_rate)
            total_steps = len(dataloader) * num_epochs_local // accumulation_steps
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )
            
            for epoch in range(num_epochs_local):
                loss = dp_client.local_training_with_dp(
                    dataloader, optimizer, scheduler, device, accumulation_steps
                )
            client_losses.append(loss)
            print(f"Client {client_idx + 1} Loss: {loss:.4f}")
        
        # Aggregate model
        server.aggregate_models()
        
        # evaluate
        test_loss, accuracy = evaluate_model(global_model, validation_dataloader, device)
        print(f"Global Model Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
