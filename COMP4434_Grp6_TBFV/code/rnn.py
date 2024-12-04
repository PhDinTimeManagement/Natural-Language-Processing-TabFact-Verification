# Define RNN Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class TableStatementRNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5
    ):
        """
        Initializes the RNN-based classifier.

        Args:
            embedding_dim (int): Dimension of the input embeddings.
            hidden_dim (int): Dimension of the hidden state in LSTM.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            bidirectional (bool, optional): If True, use a bidirectional LSTM. Defaults to False.
            dropout (float, optional): Dropout rate between LSTM layers. Defaults to 0.5.
        """
        super(TableStatementRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM for table embeddings
        self.table_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # LSTM for statement embeddings
        self.statement_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Calculate the combined hidden dimension
        combined_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        combined_hidden_dim *= 2  # Because we have two LSTMs

        # Fully connected layers
        self.fc1 = nn.Linear(combined_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, table_embeddings, statement_embeddings):
        """
        Forward pass of the model.

        Args:
            table_embeddings (torch.Tensor): Tensor of shape (batch_size, table_seq_len, embedding_dim).
            statement_embeddings (torch.Tensor): Tensor of shape (batch_size, statement_seq_len, embedding_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        # Pass table embeddings through LSTM
        table_output, (table_hidden, table_cell) = self.table_lstm(table_embeddings)
        if self.bidirectional:
            table_hidden = torch.cat((table_hidden[-2], table_hidden[-1]), dim=1)
        else:
            table_hidden = table_hidden[-1]

        # Pass statement embeddings through LSTM
        statement_output, (statement_hidden, statement_cell) = self.statement_lstm(statement_embeddings)
        if self.bidirectional:
            statement_hidden = torch.cat((statement_hidden[-2], statement_hidden[-1]), dim=1)
        else:
            statement_hidden = statement_hidden[-1]

        # Concatenate the final hidden states
        combined = torch.cat((table_hidden, statement_hidden), dim=1)

        # Pass through fully connected layers
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # For binary classification, it's common to output logits and apply sigmoid later
        return x

# Example Usage
if __name__ == "__main__":
    # Sample dimensions
    batch_size = 32
    table_seq_len = 10
    statement_seq_len = 15
    embedding_dim = 100
    hidden_dim = 128
    num_layers = 2
    bidirectional = True
    dropout = 0.3

    # Instantiate the model
    model = TableStatementRNN(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout
    )

    # Sample input tensors
    table_embeddings = torch.randn(batch_size, table_seq_len, embedding_dim)
    statement_embeddings = torch.randn(batch_size, statement_seq_len, embedding_dim)

    # Forward pass
    logits = model(table_embeddings, statement_embeddings)

    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits).squeeze()

    # Binary predictions
    predictions = (probabilities >= 0.5).long()

    print(f"Logits shape: {logits.shape}")           # Expected: (batch_size, 1)
    print(f"Probabilities shape: {probabilities.shape}")  # Expected: (batch_size)
    print(f"Predictions shape: {predictions.shape}")      # Expected: (batch_size)