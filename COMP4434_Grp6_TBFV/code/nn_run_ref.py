'''
#* Project requirement 5 :
Apply one basic neural network, such as multi-layer perceptron, CNNs, and RNNs to the same
dataset. Summarize the performance of traditional machine learning algorithms, the basic neural
network, and the proposed framework, in a table.
'''

# Import Libraries
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Added for data splitting
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json
vocab = json.load(open('vocab.json'))

# Check if CUDA is available
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {DEVICE}')

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
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# 3. Model Definition
# -------------------------------

class TextClassifierCustom(nn.Module):
    """
    BERT-based classifier for text data.
    """
    def __init__(self, num_labels=2):
        super(TextClassifierCustom, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the BERT model and classifier.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:,0]  # CLS token
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
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
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

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
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
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
    BATCH_SIZE = 16
    EPOCHS = 8
    MODEL_NAME = 'TextClassifierCustom'  # Define the model name
    MODEL_SAVE_PATH = f'../models/{MODEL_NAME}.bin'
    RESULTS_DIR = '../nn_runs'
    RESULTS_FILE = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_results.csv")

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

    # Initialize Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create Dataset and DataLoader for Training
    train_dataset = TextDatasetCustom(
        texts=X_train,
        labels=y_train,
        tokenizer=tokenizer,
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
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    # Initialize Model
    model = TextClassifierCustom(num_labels=2)
    model = model.to(DEVICE)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

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

    best_val_accuracy = 0
    best_model_state = None

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

        # Check if this is the best model so far
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict()
            print(f'New best model found at epoch {epoch +1} with validation accuracy {best_val_accuracy:.4f}')

    # Save the best model
    if best_model_state is not None:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(best_model_state, MODEL_SAVE_PATH)
        print(f"\nBest model saved to {MODEL_SAVE_PATH} with validation accuracy {best_val_accuracy:.4f}")
    else:
        print("\nNo improvement during training. Saving the last model state.")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    # Load the best model for testing
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
            tokenizer=tokenizer,
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