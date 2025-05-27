import numpy as np
from typing import List, Tuple, Dict
from neural_network import TraitPredictor

class GeneticOptimizer:
    def __init__(self, predictor: TraitPredictor, target_traits: np.ndarray):
        self.predictor = predictor
        self.target_traits = target_traits
        self.nucleotides = np.array([0, 1, 2, 3])  # A, G, C, T
        
    def optimize_sequence(self, sequence: np.ndarray, 
                         max_iterations: int = 1000,
                         mutation_rate: float = 0.1) -> Tuple[np.ndarray, float]:
        """Optimize a genetic sequence to match target traits."""
        current_sequence = sequence.copy()
        current_traits = self.predictor.predict_traits(current_sequence)
        current_fitness = self._calculate_fitness(current_traits)
        
        best_sequence = current_sequence.copy()
        best_fitness = current_fitness
        
        for _ in range(max_iterations):
            # Create a mutated version of the sequence
            mutated_sequence = self._mutate_sequence(current_sequence, mutation_rate)
            mutated_traits = self.predictor.predict_traits(mutated_sequence)
            mutated_fitness = self._calculate_fitness(mutated_traits)
            
            # Update if mutation is better
            if mutated_fitness > current_fitness:
                current_sequence = mutated_sequence
                current_fitness = mutated_fitness
                
                # Update best if current is better
                if current_fitness > best_fitness:
                    best_sequence = current_sequence.copy()
                    best_fitness = current_fitness
        
        return best_sequence, best_fitness
    
    def _mutate_sequence(self, sequence: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Apply random mutations to the sequence."""
        mutated = sequence.copy()
        mutation_mask = np.random.random(len(sequence)) < mutation_rate
        
        # For positions that will be mutated, randomly choose new nucleotides
        new_nucleotides = np.random.choice(self.nucleotides, size=np.sum(mutation_mask))
        mutated[mutation_mask] = new_nucleotides
        
        return mutated
    
    def _calculate_fitness(self, predicted_traits: np.ndarray) -> float:
        """Calculate how well the predicted traits match the target traits."""
        # Using mean squared error as fitness metric
        return -np.mean((predicted_traits - self.target_traits) ** 2)
    
    def optimize_multiple_sequences(self, sequences: Dict[str, np.ndarray],
                                  max_iterations: int = 1000,
                                  mutation_rate: float = 0.1) -> Dict[str, np.ndarray]:
        """Optimize multiple genetic sequences."""
        optimized_sequences = {}
        
        for name, sequence in sequences.items():
            try:
                optimized_sequence, fitness = self.optimize_sequence(
                    sequence,
                    max_iterations=max_iterations,
                    mutation_rate=mutation_rate
                )
                optimized_sequences[name] = optimized_sequence
                print(f"Optimized sequence {name} with fitness: {fitness}")
            except Exception as e:
                print(f"Error optimizing sequence {name}: {str(e)}")
                # Keep original sequence if optimization fails
                optimized_sequences[name] = sequence
        
        return optimized_sequences 