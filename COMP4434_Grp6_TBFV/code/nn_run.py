import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Class
class QqpDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_length=512, build_vocab=False):
        self.texts_a = []
        self.texts_b = []
        self.labels = []
        self.max_length = max_length
        self.vocab = vocab if vocab else defaultdict(int)
        self.build_vocab = build_vocab

        self._load_data(data_path)

        if build_vocab:
            self._build_vocab()
        self._encode_texts()

    def _load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 6:
                    continue  # Skip malformed rows
                _, _, _, context, question, label = row
                self.texts_a.append(context)
                self.texts_b.append(question)
                self.labels.append(int(label))

    def _build_vocab(self):
        for text in self.texts_a + self.texts_b:
            for word in text.lower().split():
                self.vocab[word] += 1
        # Initialize with <PAD> and <UNK>
        self.vocab = {word: idx+2 for idx, (word, count) in enumerate(self.vocab.items()) if count >=1}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1

    def _encode_texts(self):
        self.encoded_a = [self._encode(text) for text in self.texts_a]
        self.encoded_b = [self._encode(text) for text in self.texts_b]

    def _encode(self, text):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in text.lower().split()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a = self.encoded_a[idx]
        b = self.encoded_b[idx]
        label = self.labels[idx]
        return a, b, label

# Collate Function
def collate_fn(batch):
    a_batch, b_batch, labels = zip(*batch)
    max_len_a = max(len(a) for a in a_batch)
    max_len_b = max(len(b) for b in b_batch)
    max_len = max(max_len_a, max_len_b, 1)

    padded_a = [a + [0]*(max_len - len(a)) for a in a_batch]
    padded_b = [b + [0]*(max_len - len(b)) for b in b_batch]

    return {
        'text_a': torch.tensor(padded_a, dtype=torch.long),
        'text_b': torch.tensor(padded_b, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

# DataLoader Function
def get_data_loaders(data_path, batch_size=16, test_size=0.2, max_length=512):
    # Initialize dataset and build vocabulary
    dataset = QqpDataset(data_path, build_vocab=True, max_length=max_length)
    vocab = dataset.vocab

    # Split into train and test
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, vocab

# RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=2, n_layers=2, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_a, text_b):
        embedded_a = self.embedding(text_a)  # [batch_size, seq_len_a, embed_dim]
        embedded_b = self.embedding(text_b)  # [batch_size, seq_len_b, embed_dim]
        embedded = torch.cat((embedded_a, embedded_b), dim=1)  # [batch_size, seq_len_a + seq_len_b, embed_dim]

        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))  # [batch_size, hidden_dim*2]
        output = self.fc(hidden)  # [batch_size, output_dim]
        return output

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_val_accuracy = 0
    results = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total = 0

        for batch in train_loader:
            text_a = batch['text_a'].to(device)
            text_b = batch['text_b'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(text_a, text_b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = epoch_loss / total
        train_accuracy = epoch_correct / total

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}')

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(SCRIPT_DIR, 'best_rnn_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model to {best_model_path}")

    return results

# Evaluation Function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            text_a = batch['text_a'].to(device)
            text_b = batch['text_b'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(text_a, text_b)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    loss = epoch_loss / total
    accuracy = epoch_correct / total
    return loss, accuracy

# Save Results Function
def save_results(results, csv_path='training_results.csv'):
    # Update csv_path to be under SCRIPT_DIR
    csv_full_path = os.path.join(SCRIPT_DIR, csv_path)
    keys = results[0].keys()
    with open(csv_full_path, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f'Results saved to {csv_full_path}')

# Load Model Function
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Main Function
def main():
    data_path = '../dataset/tsv_data_horizontal/simple_test.tsv'
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3
    max_length = 512

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders and vocabulary
    train_loader, test_loader, vocab = get_data_loaders(data_path, batch_size=batch_size, test_size=0.2, max_length=max_length)
    vocab_size = len(vocab)
    print(f'Vocabulary Size: {vocab_size}')

    # Initialize model, criterion, optimizer
    model = RNNClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, output_dim=2, n_layers=2, dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    results = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs)

    # Save training results
    save_results(results, csv_path='training_results.csv')

    # Load the best model for testing
    best_model = RNNClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, output_dim=2, n_layers=2, dropout=0.5)
    best_model_path = os.path.join(SCRIPT_DIR, 'best_rnn_model.pth')
    best_model = load_model(best_model, best_model_path, device)

    # Evaluate on test set
    test_loss, test_accuracy = evaluate_model(best_model, test_loader, criterion, device)
    print(f'Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.4f}')

    # Save test results
    test_results_path = os.path.join(SCRIPT_DIR, 'test_results.csv')
    with open(test_results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Test Loss', 'Test Accuracy'])
        writer.writerow([test_loss, test_accuracy])
    print(f'Test results saved to {test_results_path}')

if __name__ == "__main__":
    main()