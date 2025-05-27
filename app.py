from flask import Flask, render_template, request, send_file, jsonify
import os
import pandas as pd
import numpy as np
from genetic_parser import GeneticParser
from neural_network import TraitPredictor
from optimizer import GeneticOptimizer
import tempfile
import traceback

app = Flask(__name__)

# Initialize components
genetic_parser = GeneticParser()

# Store temporary files
temp_files = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_input.name)
        temp_input.close()
        
        # Read sequences and validate
        sequences, problems = genetic_parser.read_csv(temp_input.name)
        
        if not sequences:
            return jsonify({'error': 'No valid sequences found in the file'}), 400
            
        # Get the maximum sequence length
        max_length = max(len(seq) for seq in sequences.values())
        
        # Initialize predictor with the maximum sequence length
        predictor = TraitPredictor(sequence_length=max_length, num_traits=5)
        
        # Generate random target traits
        target_traits = np.random.random(5)
        
        # Initialize optimizer and optimize sequences
        optimizer = GeneticOptimizer(predictor, target_traits)
        optimized_sequences = optimizer.optimize_multiple_sequences(sequences)
        
        # Save optimized sequences
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        genetic_parser.write_csv(temp_output.name, optimized_sequences)
        temp_output.close()
        
        # Store the temporary file path
        file_id = os.path.basename(temp_output.name)
        temp_files[file_id] = temp_output.name
        
        # Get trait predictions for original sequences
        original_traits = {}
        for name, sequence in sequences.items():
            try:
                traits = predictor.predict_traits(sequence)
                original_traits[name] = traits.tolist()
            except Exception as e:
                print(f"Error predicting traits for sequence {name}: {str(e)}")
                continue
        
        # Get trait predictions for optimized sequences
        optimized_traits = {}
        for name, sequence in optimized_sequences.items():
            try:
                traits = predictor.predict_traits(sequence)
                optimized_traits[name] = traits.tolist()
            except Exception as e:
                print(f"Error predicting traits for optimized sequence {name}: {str(e)}")
                continue
        
        # Clean up temporary input file
        os.unlink(temp_input.name)
        
        return jsonify({
            'original_traits': original_traits,
            'optimized_traits': optimized_traits,
            'problems': problems,
            'file_id': file_id
        })
        
    except Exception as e:
        print("Error details:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/<file_id>')
def download_file(file_id):
    try:
        if file_id not in temp_files:
            return jsonify({'error': 'File not found'}), 404
            
        file_path = temp_files[file_id]
        response = send_file(
            file_path,
            as_attachment=True,
            download_name='optimized_sequences.csv',
            mimetype='text/csv'
        )
        
        # Clean up the temporary file after sending
        @response.call_on_close
        def cleanup():
            try:
                os.unlink(file_path)
                del temp_files[file_id]
            except Exception as e:
                print(f"Error cleaning up file: {str(e)}")
        
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 