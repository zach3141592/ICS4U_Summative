# Genetic Code Optimizer

A web application that analyzes and optimizes genetic sequences using neural networks. The application can identify problems in genetic code, fix them automatically, and optimize the sequences for desired traits.

## Features

- Upload genetic sequences via CSV file
- Automatic validation of genetic sequences
- Detection of common problems:
  - Invalid nucleotides
  - Missing start/stop codons
  - Incorrect sequence length
  - Premature stop codons
- Automatic fixing of identified problems
- Neural network-based trait prediction
- Sequence optimization
- Download of optimized sequences

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/genetic-code-optimizer.git
cd genetic-code-optimizer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and navigate to:

```
http://127.0.0.1:5000
```

## Input Format

The application expects a CSV file with the following columns:

- `id`: Unique identifier for each sequence
- `sequence`: Genetic sequence (containing A, G, C, T nucleotides)

Example:

```csv
id,sequence
seq1,ATGCGTACGTAGTAA
seq2,ATGCGTACGTTAG
```

## Project Structure

- `app.py`: Flask web application
- `genetic_parser.py`: Genetic sequence parsing and validation
- `neural_network.py`: Neural network model for trait prediction
- `optimizer.py`: Genetic sequence optimization
- `templates/index.html`: Web interface
- `requirements.txt`: Project dependencies

## Requirements

- Python 3.7+
- Flask
- NumPy
- Pandas
- scikit-learn

## License

MIT License
