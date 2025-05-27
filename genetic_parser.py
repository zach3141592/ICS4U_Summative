import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

class SequenceStack:
    """A stack to track sequence processing history."""
    def __init__(self):
        self.stack = []
        self.processing_history = {}
    
    def push(self, sequence_id: str, sequence: np.ndarray, problems: List[str]) -> None:
        """Push a sequence onto the stack with its problems."""
        self.stack.append((sequence_id, sequence, problems))
        self.processing_history[sequence_id] = {
            'original_problems': problems.copy(),
            'optimization_steps': []
        }
    
    def pop(self) -> Optional[Tuple[str, np.ndarray, List[str]]]:
        """Pop a sequence from the stack."""
        return self.stack.pop() if self.stack else None
    
    def peek(self) -> Optional[Tuple[str, np.ndarray, List[str]]]:
        """Peek at the top sequence without removing it."""
        return self.stack[-1] if self.stack else None
    
    def add_optimization_step(self, sequence_id: str, step: str) -> None:
        """Add an optimization step to the sequence's history."""
        if sequence_id in self.processing_history:
            self.processing_history[sequence_id]['optimization_steps'].append(step)
    
    def get_history(self, sequence_id: str) -> Dict:
        """Get the processing history for a sequence."""
        return self.processing_history.get(sequence_id, {})
    
    def is_empty(self) -> bool:
        """Check if the stack is empty."""
        return len(self.stack) == 0
    
    def size(self) -> int:
        """Get the size of the stack."""
        return len(self.stack)

class GeneticParser:
    def __init__(self):
        self.nucleotides = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
        self.reverse_nucleotides = {v: k for k, v in self.nucleotides.items()}
        self.start_codon = 'ATG'
        self.stop_codons = ['TAA', 'TAG', 'TGA']
        self.sequence_stack = SequenceStack()
        
    def parse_sequence(self, sequence: str) -> np.ndarray:
        """Convert a DNA sequence to numerical representation."""
        sequence = sequence.upper()
        if not all(n in self.nucleotides for n in sequence):
            raise ValueError("Invalid nucleotide in sequence")
        return np.array([self.nucleotides[n] for n in sequence])
    
    def sequence_to_text(self, sequence: np.ndarray) -> str:
        """Convert numerical representation back to DNA sequence."""
        return ''.join(self.reverse_nucleotides[n] for n in sequence)
    
    def validate_sequence(self, sequence: str) -> List[str]:
        """Validate a genetic sequence and return any problems found."""
        problems = []
        
        # Check for invalid nucleotides
        invalid_chars = [c for c in sequence if c not in self.nucleotides]
        if invalid_chars:
            problems.append(f"Invalid nucleotides found: {', '.join(set(invalid_chars))}")
        
        # Check sequence length
        if len(sequence) % 3 != 0:
            problems.append("Sequence length is not a multiple of 3 (incomplete codons)")
        
        # Check for start codon
        if not sequence.startswith(self.start_codon):
            problems.append("Missing start codon (ATG)")
        
        # Check for stop codon
        has_stop = any(sequence.endswith(stop) for stop in self.stop_codons)
        if not has_stop:
            problems.append("Missing stop codon (TAA, TAG, or TGA)")
        
        # Check for premature stop codons
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        for i, codon in enumerate(codons[:-1]):  # Exclude last codon
            if codon in self.stop_codons:
                problems.append(f"Premature stop codon found at position {i*3}")
        
        return problems
    
    def fix_sequence(self, sequence: str) -> str:
        """Fix common problems in a genetic sequence."""
        sequence = sequence.upper()
        original_sequence = sequence
        
        # Remove invalid nucleotides
        sequence = ''.join(c for c in sequence if c in self.nucleotides)
        if sequence != original_sequence:
            self.sequence_stack.add_optimization_step('current', 'Removed invalid nucleotides')
        
        # Ensure length is multiple of 3
        remainder = len(sequence) % 3
        if remainder != 0:
            sequence = sequence[:-remainder]
            self.sequence_stack.add_optimization_step('current', 'Adjusted sequence length to multiple of 3')
        
        # Add start codon if missing
        if not sequence.startswith(self.start_codon):
            sequence = self.start_codon + sequence
            self.sequence_stack.add_optimization_step('current', 'Added start codon')
        
        # Add stop codon if missing
        if not any(sequence.endswith(stop) for stop in self.stop_codons):
            sequence = sequence + self.stop_codons[0]
            self.sequence_stack.add_optimization_step('current', 'Added stop codon')
        
        return sequence
    
    def bubble_sort_sequences(self, sequences: Dict[str, np.ndarray], 
                            problems: Dict[str, List[str]]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        """Sort sequences by number of issues using bubble sort."""
        # Convert to list of tuples for sorting
        seq_list = [(id, seq, problems.get(id, [])) for id, seq in sequences.items()]
        
        # Bubble sort
        n = len(seq_list)
        for i in range(n):
            for j in range(0, n-i-1):
                if len(seq_list[j][2]) < len(seq_list[j+1][2]):
                    seq_list[j], seq_list[j+1] = seq_list[j+1], seq_list[j]
                    self.sequence_stack.add_optimization_step('sorting', f'Swapped sequences {seq_list[j][0]} and {seq_list[j+1][0]}')
        
        # Convert back to dictionaries
        sorted_sequences = {id: seq for id, seq, _ in seq_list}
        sorted_problems = {id: probs for id, _, probs in seq_list}
        
        return sorted_sequences, sorted_problems
    
    def read_csv(self, filename: str, sequence_column: str = 'sequence', 
                id_column: str = 'id') -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        """Read genetic sequences from a CSV file and validate them."""
        try:
            # Read CSV file
            df = pd.read_csv(filename)
            
            # Validate required columns
            if sequence_column not in df.columns:
                raise ValueError(f"Column '{sequence_column}' not found in CSV file")
            if id_column not in df.columns:
                raise ValueError(f"Column '{id_column}' not found in CSV file")
            
            # Convert sequences to numerical format and validate
            sequences = {}
            problems = {}
            
            for _, row in df.iterrows():
                sequence_id = str(row[id_column])
                sequence = str(row[sequence_column])
                
                # Validate sequence
                sequence_problems = self.validate_sequence(sequence)
                if sequence_problems:
                    problems[sequence_id] = sequence_problems
                    # Fix sequence
                    fixed_sequence = self.fix_sequence(sequence)
                    sequences[sequence_id] = self.parse_sequence(fixed_sequence)
                    # Push to stack
                    self.sequence_stack.push(sequence_id, sequences[sequence_id], sequence_problems)
                else:
                    sequences[sequence_id] = self.parse_sequence(sequence)
                    # Push to stack
                    self.sequence_stack.push(sequence_id, sequences[sequence_id], [])
            
            # Sort sequences by number of issues
            sequences, problems = self.bubble_sort_sequences(sequences, problems)
            
            return sequences, problems
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file")
    
    def write_csv(self, filename: str, sequences: Dict[str, np.ndarray],
                 sequence_column: str = 'sequence', id_column: str = 'id'):
        """Write genetic sequences to a CSV file."""
        # Convert sequences to text format
        data = {
            id_column: list(sequences.keys()),
            sequence_column: [self.sequence_to_text(seq) for seq in sequences.values()]
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def get_codons(self, sequence: np.ndarray) -> List[str]:
        """Extract codons from a sequence."""
        text_sequence = self.sequence_to_text(sequence)
        return [text_sequence[i:i+3] for i in range(0, len(text_sequence), 3)]
    
    def get_sequence_history(self, sequence_id: str) -> Dict:
        """Get the processing history for a sequence."""
        return self.sequence_stack.get_history(sequence_id) 