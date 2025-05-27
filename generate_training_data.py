import numpy as np

# Parameters
sequence_length = 21  # Example length
num_traits = 5
num_examples = 1000
output_file = 'training_data.npz'

# Set random seed for reproducibility
np.random.seed(42)

# Initialize arrays for sequences and traits
X = np.zeros((num_examples, sequence_length))
y = np.zeros((num_examples, num_traits))

print(f"Initializing arrays with shapes:")
print(f"X: {X.shape} (num_examples x sequence_length)")
print(f"y: {y.shape} (num_examples x num_traits)")

# Define patterns as lists of integers
patterns = [
    [1, 2, 3],  # GCT
    [0, 3, 2],  # ATC
    [3, 0, 0],  # TAA
]

for i in range(num_examples):
    # Generate a sequence with known patterns
    sequence = np.zeros(sequence_length)
    sequence[:3] = np.array([0, 3, 1])  # ATG
    for j in range(3, len(sequence) - 3, 3):
        pattern = patterns[np.random.randint(len(patterns))]
        sequence[j:j+3] = pattern
    sequence[-3:] = np.array([3, 0, 0])  # TAA

    # Generate trait values
    traits = np.zeros(num_traits)
    high_expr_count = np.sum(sequence == 1)  # G's
    traits[0] = high_expr_count / sequence_length
    at_content = np.sum((sequence == 0) | (sequence == 3))
    gc_content = np.sum((sequence == 1) | (sequence == 2))
    traits[1] = 1 - abs(at_content - gc_content) / sequence_length
    stop_codon_count = np.sum(sequence == 3)  # T's
    traits[2] = stop_codon_count / sequence_length
    traits[3] = high_expr_count / sequence_length
    low_expr_count = np.sum(sequence == 2)  # C's
    traits[4] = low_expr_count / sequence_length
    traits = np.clip(traits, 0, 1)

    X[i] = sequence
    y[i] = traits

# Save as NPZ file
np.savez(output_file, X=X, y=y)
print(f"\nGenerated {output_file} with {num_examples} examples.")
print(f"Final shapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}") 