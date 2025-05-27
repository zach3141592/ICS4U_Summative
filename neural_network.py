import numpy as np
from typing import List, Tuple

class TraitPredictor:
    def __init__(self, sequence_length: int, num_traits: int):
        self.sequence_length = sequence_length
        self.num_traits = num_traits
        # Initialize weights with correct dimensions: (sequence_length * 4, num_traits)
        self.weights = np.random.randn(sequence_length * 4, num_traits) * 0.01
        self.bias = np.zeros(num_traits)
        
    def one_hot_encode(self, sequence: np.ndarray) -> np.ndarray:
        """Convert numerical sequence to one-hot encoding."""
        # Pad sequence if shorter than expected length
        if len(sequence) < self.sequence_length:
            padded_sequence = np.pad(sequence, (0, self.sequence_length - len(sequence)), 
                                   mode='constant', constant_values=0)
        else:
            padded_sequence = sequence
            
        # Create one-hot encoding
        one_hot = np.zeros((self.sequence_length, 4))
        one_hot[np.arange(self.sequence_length), padded_sequence] = 1
        # Flatten the one-hot encoding
        return one_hot.flatten()
    
    def predict_traits(self, sequence: np.ndarray) -> np.ndarray:
        """Predict traits from a genetic sequence."""
        # One-hot encode the sequence
        encoded = self.one_hot_encode(sequence)
        
        # Simple linear model with sigmoid activation
        z = np.dot(encoded, self.weights) + self.bias
        predictions = 1 / (1 + np.exp(-z))  # sigmoid activation
        
        return predictions
    
    def train(self, sequences: List[np.ndarray], traits: List[np.ndarray], 
              epochs: int = 10, learning_rate: float = 0.01):
        """Train the model on genetic sequences and their traits."""
        for epoch in range(epochs):
            total_loss = 0
            for sequence, target_traits in zip(sequences, traits):
                # Forward pass
                predictions = self.predict_traits(sequence)
                
                # Calculate loss (binary cross-entropy)
                loss = -np.mean(target_traits * np.log(predictions + 1e-15) + 
                              (1 - target_traits) * np.log(1 - predictions + 1e-15))
                total_loss += loss
                
                # Backward pass
                encoded = self.one_hot_encode(sequence)
                error = predictions - target_traits
                
                # Update weights and bias
                self.weights -= learning_rate * np.outer(encoded, error)
                self.bias -= learning_rate * error
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(sequences)}")
    
    def save_model(self, filepath: str):
        """Save the model weights and bias."""
        np.savez(filepath, weights=self.weights, bias=self.bias)
    
    def load_model(self, filepath: str):
        """Load the model weights and bias."""
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias'] 