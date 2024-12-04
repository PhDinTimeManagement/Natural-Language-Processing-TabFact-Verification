'''
#* Project requirement 5 :
Apply one basic neural network, such as multi-layer perceptron, CNNs, and RNNs to the same
dataset. Summarize the performance of traditional machine learning algorithms, the basic neural
network, and the proposed framework, in a table.
'''

from datasets import load_dataset
import pandas as pd
import torch
import transformers
from transformers import BertTokenizer, BertModel

# Load the TabFact dataset
dataset = load_dataset("wenhu/tab_fact", name="tab_fact") # Load the dataset with this huggingface link

# Preprocess the dataset
#* Transform the dataset so it contains the tokenized table in a single string, statement, and label
def parse_table(table_text):
    """
    Parse the table_text into a pandas DataFrame.
    """
    rows = table_text.strip().split("\n")
    headers = rows[0].split("#")
    data = [row.split("#") for row in rows[1:]]
    return pd.DataFrame(data, columns=headers)

def serialize_table(table, table_caption):
    """
    Serialize the table as a single string with the table caption.
    """
    serialized_rows = []
    for _, row in table.iterrows():
        serialized_row = ", ".join(f"{col}: {row[col]}" for col in table.columns)
        serialized_rows.append(serialized_row)
    serialized_table = " ".join(serialized_rows)
    return f"{table_caption} {serialized_table}"

def preprocess_data(dataset):
    """
    Preprocess the dataset to return a list of rows with tokenized table, statement, and label.
    """
    processed_dataset = []
    
    for row in dataset:
        # Parse and serialize the table
        table = parse_table(row["table_text"])
        serialized_table = serialize_table(table, row["table_caption"])
        
        # Extract statement and label
        statement = row["statement"]
        label = row["label"]
        
        # Append processed row
        processed_dataset.append({
            "table": serialized_table,
            "statement": statement,
            "label": label
        })
    
    return processed_dataset

#! For the ML model, we can feed the embeddings of the table and statement into the model
# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
transformers.logging.set_verbosity_error() # Suppress the annoying warnings

def extract_embeddings(row, tokenizer, model, max_len=512):
    """
    Extract embeddings for table and statement, and return a dictionary with embeddings.
    """
    table = row["table"]
    statement = row["statement"]

    # Tokenize the inputs
    inputs = tokenizer(
        table,
        statement,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings: Mean over token embeddings for a fixed-size representation
    table_statement_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

    # Return a dictionary with the embeddings replacing table and statement
    return {
        "table": table_statement_embedding,
        "statement": table_statement_embedding,
        "label": row["label"]
    }

def extract_embeddings_for_dataset(dataset, tokenizer, model, max_len=512):
    """
    Extract embeddings for the entire dataset.
    """
    return [extract_embeddings(row, tokenizer, model, max_len) for row in dataset]

# Extract splits
train_dataset = preprocess_data(dataset["train"])
val_dataset = preprocess_data(dataset["validation"])
test_dataset = preprocess_data(dataset["test"])

# Extract embeddings for the entire dataset
train_dataset_embeddings = extract_embeddings_for_dataset(train_dataset, tokenizer, model)
val_dataset_embeddings = extract_embeddings_for_dataset(val_dataset, tokenizer, model)
test_dataset_embeddings = extract_embeddings_for_dataset(test_dataset, tokenizer, model)

print(train_dataset[0])
print(train_dataset_embeddings[0])

from rnn import TableStatementRNN
import torch.nn as nn

model = TableStatementRNN(embedding_dim=512, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.2)

# Train the model
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_dataset_embeddings, val_dataset_embeddings, device, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_dataset_embeddings, dtype=torch.float32).to(device)
    val_data = torch.tensor(val_dataset_embeddings, dtype=torch.float32).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store metrics for CSV
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = torch.tensor([item["label"] for item in train_dataset_embeddings[i:i + batch_size]], 
                                     dtype=torch.float32).to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_data["table"], batch_data["statement"])
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i + batch_size]
                batch_labels = torch.tensor([item["label"] for item in val_dataset_embeddings[i:i + batch_size]], 
                                         dtype=torch.float32).to(device)
                
                outputs = model(batch_data["table"], batch_data["statement"])
                val_loss += criterion(outputs.squeeze(), batch_labels).item()
                
                predictions = (torch.sigmoid(outputs.squeeze()) >= 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        # Calculate epoch statistics
        avg_train_loss = total_loss / (len(train_data) / batch_size)
        avg_val_loss = val_loss / (len(val_data) / batch_size)
        val_accuracy = correct / total
        
        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    
    # Save metrics to CSV
    df = pd.DataFrame(metrics)
    df.to_csv('training_metrics.csv', index=False)
    
    return model

load_model = 'model.pth'
model.load_state_dict(torch.load(load_model))

# Evaluate the model





# Test the model


# Summarize the performance of traditional machine learning algorithms, the basic neural network, and the proposed framework, in a table in a csv file

