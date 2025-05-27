"""Name: Zach Yu
Class: ICS4U
Date: May 26, 2025
Teacher: Mr. K
Description: This is a simple web app that takes in a csv file input of genetic code, 
identifies the issues in the genetic code, and uses machine learning to optimize the code.
This new code is then outputted as a csv file.
"""
from flask import Flask, render_template, request, send_file, jsonify
import os
import pandas as pd #for file handling and data manipulation
import numpy as np
from genetic_parser import GeneticParser #instanciate objects
from neural_network import TraitPredictor  #instanciate objects
from optimizer import GeneticOptimizer #instanciate objects
import tempfile #temporary file storage
import traceback #error handling
#dependencies

app = Flask(__name__)

# initialize genetic parser object
genetic_parser = GeneticParser()

# Store temporary files
temp_files = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_input.name)
        temp_input.close()
        
        # Parse and validate sequences
        sequences, problems = genetic_parser.read_csv(temp_input.name)
        
        # Find sequences with most and least issues
        if problems:
            most_issues = max(problems.items(), key=lambda x: len(x[1]))
            least_issues = min(problems.items(), key=lambda x: len(x[1]))
            issues_info = {
                'most_issues': {
                    'id': most_issues[0],
                    'count': len(most_issues[1]),
                    'problems': most_issues[1]
                },
                'least_issues': {
                    'id': least_issues[0],
                    'count': len(least_issues[1]),
                    'problems': least_issues[1]
                }
            }
        else:
            issues_info = {
                'most_issues': None,
                'least_issues': None
            }
        
        # Create trait predictor and optimizer
        predictor = TraitPredictor(sequence_length=len(next(iter(sequences.values()))), num_traits=5)
        target_traits = np.array([0.8, 0.6, 0.7, 0.9, 0.5])  # Example target traits
        optimizer = GeneticOptimizer(predictor, target_traits)
        
        # Optimize sequences
        optimized_sequences = optimizer.optimize_multiple_sequences(sequences)
        
        # Save optimized sequences to temporary file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        genetic_parser.write_csv(temp_output.name, optimized_sequences)
        temp_output.close()
        
        # Store the temporary file path
        file_id = os.path.basename(temp_output.name)
        temp_files[file_id] = temp_output.name
        
        # Get trait predictions for original and optimized sequences
        original_traits = {}
        optimized_traits = {}
        
        for seq_id, sequence in sequences.items():
            try:
                original_traits[seq_id] = predictor.predict_traits(sequence).tolist()
                optimized_traits[seq_id] = predictor.predict_traits(optimized_sequences[seq_id]).tolist()
            except Exception as e:
                print(f"Error predicting traits for sequence {seq_id}: {str(e)}")
                original_traits[seq_id] = []
                optimized_traits[seq_id] = []
        
        # Clean up input file
        os.unlink(temp_input.name)
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'problems': problems,
            'issues_info': issues_info,
            'original_traits': original_traits,
            'optimized_traits': optimized_traits
        })
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
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