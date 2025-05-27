import numpy as np
from typing import List, Tuple, Dict
import os

class TraitPredictor:
    def __init__(self, sequence_length: int, num_traits: int):
        self.sequence_length = sequence_length
        self.num_traits = num_traits
        self.learning_rate = 0.01
        
        # Initialize weights with correct dimensions (sequence_length x num_traits)
        self.weights = np.random.randn(self.sequence_length, self.num_traits) * 0.01
        print(f"Initialized weights with shape: {self.weights.shape}")
        
        # Only load training data from file
        self.training_data = self._load_training_data()
        print(f"Loaded training data shapes - X: {self.training_data[0].shape}, y: {self.training_data[1].shape}")
    
    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from file. Raise error if not found."""
        data_file = 'training_data.npz'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data file '{data_file}' not found. Please provide it.")
        data = np.load(data_file)
        return data['X'], data['y']
    
    def train(self, epochs: int = 1000, batch_size: int = 32) -> List[float]:
        """Train the neural network on the generated data."""
        X, y = self.training_data
        num_examples = len(X)
        losses = []
        
        print(f"Starting training with:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"weights shape: {self.weights.shape}")
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(num_examples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, num_examples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                predictions = np.dot(batch_X, self.weights)
                
                # Calculate loss
                loss = np.mean((predictions - batch_y) ** 2)
                epoch_loss += loss
                
                # Backward pass
                error = predictions - batch_y
                gradient = np.dot(batch_X.T, error) / batch_size
                
                # Update weights
                self.weights -= self.learning_rate * gradient
            
            # Record average loss for the epoch
            losses.append(epoch_loss / (num_examples / batch_size))
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict_traits_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Predict traits for a batch of sequences."""
        return np.dot(sequences, self.weights)
    
    def predict_traits(self, sequence: np.ndarray) -> np.ndarray:
        """Predict traits for a single sequence."""
        # Ensure sequence is 1-dimensional
        if len(sequence.shape) > 1:
            sequence = sequence.flatten()
        # Make prediction
        return np.dot(sequence, self.weights)
    
    def one_hot_encode(self, sequence: np.ndarray) -> np.ndarray:
        """Convert numerical sequence to one-hot encoding."""
        # Ensure sequence is 1-dimensional
        if len(sequence.shape) > 1:
            sequence = sequence.flatten()
        
        encoded = np.zeros((len(sequence), 4))
        for i, n in enumerate(sequence):
            encoded[i, int(n)] = 1
        return encoded.flatten()
    
    def save_weights(self, filename: str):
        """Save the trained weights to a file."""
        np.save(filename, self.weights)
    
    def load_weights(self, filename: str):
        """Load weights from a file."""
        self.weights = np.load(filename)

    def save_model(self, filepath: str):
        """Save the model weights and bias."""
        np.savez(filepath, weights=self.weights)
    
    def load_model(self, filepath: str):
        """Load the model weights and bias."""
        data = np.load(filepath)
        self.weights = data['weights'] 