# Import Libraries
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Added for data splitting
import numpy as np
from tqdm import tqdm
import json

# Load Vocabulary
vocab = json.load(open('vocab.json'))

# Define special tokens and their indices
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_IDX = vocab.get(PAD_TOKEN, 0)
UNK_IDX = vocab.get(UNK_TOKEN, 1)

# Check if CUDA is available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {DEVICE}')

# -------------------------------
# Early Stopping Class
# -------------------------------

class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """
    def __init__(self, patience=3, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation accuracy improved.
            verbose (bool): If True, prints a message for each validation accuracy improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_accuracy = 0.0

    def __call__(self, val_accuracy, model):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model):
        '''Saves model when validation accuracy increases.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.best_accuracy:.6f} --> {val_accuracy:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_accuracy = val_accuracy

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------

class DataLoaderCustom:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        """
        Loads the TSV data into a pandas DataFrame.
        """
        df = pd.read_csv(self.filepath, sep='\t', header=None, quoting=3)
        # Assign column names based on the provided data snippet
        # Adjust the number of columns if necessary
        if df.shape[1] == 6:
            df.columns = ['id', 'number', 'category', 'description', 'sentence', 'label']
        elif df.shape[1] == 5:
            df.columns = ['id', 'number', 'category', 'sentence', 'label']
        else:
            # Handle unexpected number of columns
            df.columns = [f'col_{i}' for i in range(df.shape[1])]
        return df

    def preprocess(self, df):
        """
        Preprocesses the DataFrame by encoding categorical variables and handling missing values.
        """
        # Convert label to integer
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
        else:
            print("Warning: 'label' column not found.")
            df['label'] = 0  # Default label

        # Encode categorical 'category' column if exists
        if 'category' in df.columns:
            label_encoder = LabelEncoder()
            df['category_encoded'] = label_encoder.fit_transform(df['category'])
        else:
            df['category_encoded'] = 0  # Default encoding

        # Handle missing values by dropping
        df = df.dropna()
        return df

# -------------------------------
# 2. Dataset Class
# -------------------------------

class TextDatasetCustom(Dataset):
    """
    Custom Dataset for loading text data.
    """
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        """
        Tokenizes the input text into a list of word indices.
        """
        tokens = text.lower().split()
        indices = [self.vocab.get(token, UNK_IDX) for token in tokens]
        return indices

    def pad_sequence(self, seq):
        """
        Pads or truncates the sequence to the maximum length.
        """
        if len(seq) < self.max_len:
            seq = seq + [PAD_IDX] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return seq

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        tokens = self.tokenize(text)
        tokens_padded = self.pad_sequence(tokens)
        return {
            'input_ids': torch.tensor(tokens_padded, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# 3. Model Definition
# -------------------------------

class RNNClassifier(nn.Module):
    """
    Basic RNN-based classifier for text data.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_labels, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional

    def forward(self, input_ids):
        """
        Forward pass through the RNN model.
        """
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)     # (batch_size, seq_len, hidden_dim*2)
        # Use the mean of the LSTM outputs for classification
        lstm_out = torch.mean(lstm_out, dim=1)
        dropout_out = self.dropout(lstm_out)
        logits = self.fc(dropout_out)
        return logits

# -------------------------------
# 4. Training and Evaluation Functions
# -------------------------------

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """
    Trains the model for one epoch.
    """
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
    scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    Evaluates the model on a validation or test dataset.
    """
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# -------------------------------
# 5. Main Execution
# -------------------------------

def main():
    # File paths
    TRAIN_FILE = '../dataset/tsv_data_horizontal/train.tsv'
    TEST_FILES = [
        '../dataset/tsv_data_horizontal/test.tsv',
        '../dataset/tsv_data_horizontal/simple_test.tsv',
        '../dataset/tsv_data_horizontal/complex_test.tsv',
        '../dataset/tsv_data_horizontal/small_test.tsv'
    ]

    MAX_LEN = 160
    BATCH_SIZE = 64  # Increased batch size for RNN efficiency
    EPOCHS = 15
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_LABELS = 2
    DROPOUT = 0.3
    MODEL_NAME = 'RNNClassifier'  # Define the model name
    MODEL_SAVE_PATH = f'../models/{MODEL_NAME}.pt'  # Changed extension to .pt
    RESULTS_DIR = '../nn_runs'
    RESULTS_FILE = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_results.csv")
    EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait before stopping
    EARLY_STOPPING_VERBOSE = True  # Whether to print early stopping messages

    # Initialize DataLoader
    data_loader_custom = DataLoaderCustom(TRAIN_FILE)
    df = data_loader_custom.load_data()
    df = data_loader_custom.preprocess(df)

    X = df['sentence'].values
    y = df['label'].values

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y  # Ensures the split maintains label proportions
    )

    # Determine vocabulary size
    vocab_size = len(vocab)
    print(f'Vocabulary size: {vocab_size}')

    # Create Dataset and DataLoader for Training
    train_dataset = TextDatasetCustom(
        texts=X_train,
        labels=y_train,
        vocab=vocab,
        max_len=MAX_LEN
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Create Dataset and DataLoader for Validation
    val_dataset = TextDatasetCustom(
        texts=X_val,
        labels=y_val,
        vocab=vocab,
        max_len=MAX_LEN
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    # Initialize Model
    model = RNNClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_labels=NUM_LABELS,
        dropout=DROPOUT
    )
    model = model.to(DEVICE)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Loss function
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # Prepare to store results
    results = {
        'Epoch': [],
        'Train Loss': [],
        'Train Accuracy': [],
        'Validation Loss': [],
        'Validation Accuracy': []
    }

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=EARLY_STOPPING_VERBOSE, 
                                   delta=0.0, path=MODEL_SAVE_PATH)

    # Training Loop
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch +1}/{EPOCHS}')
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            DEVICE,
            scheduler,
            len(train_dataset)
        )
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            DEVICE,
            len(val_dataset)
        )
        print(f'Validation loss {val_loss:.4f} accuracy {val_acc:.4f}')

        # Append results
        results['Epoch'].append(epoch +1)
        results['Train Loss'].append(train_loss)
        results['Train Accuracy'].append(train_acc.item())
        results['Validation Loss'].append(val_loss)
        results['Validation Accuracy'].append(val_acc.item())

        # Check Early Stopping
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    # Load the best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model = model.to(DEVICE)

    # Prepare to store test results
    test_results = []

    # Testing on multiple test files
    for test_file in TEST_FILES:
        print(f'\nTesting on {test_file}')
        data_loader_custom_test = DataLoaderCustom(test_file)
        df_test = data_loader_custom_test.load_data()
        df_test = data_loader_custom_test.preprocess(df_test)

        X_test = df_test['sentence'].values
        y_test = df_test['label'].values

        test_dataset = TextDatasetCustom(
            texts=X_test,
            labels=y_test,
            vocab=vocab,
            max_len=MAX_LEN
        )

        test_data_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE
        )

        test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            DEVICE,
            len(test_dataset)
        )
        print(f'Test Accuracy for {test_file}: {test_acc:.4f}')

        test_results.append({
            'Test File': os.path.basename(test_file),
            'Test Accuracy': test_acc.item()
        })

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create a DataFrame for training and validation results
    results_df = pd.DataFrame(results)

    # Create a DataFrame for test results
    test_results_df = pd.DataFrame(test_results)

    # Write results to CSV
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        results_df.to_csv(csvfile, index=False)
        csvfile.write('\n')  # Add a newline to separate sections
        test_results_df.to_csv(csvfile, index=False)

    print(f"\nAll results have been saved to {RESULTS_FILE}")

if __name__ == '__main__':
    main()
