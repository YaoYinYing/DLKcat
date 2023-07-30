#!/usr/bin/python
# coding: utf-8

import os
import sys
import math
import torch
import requests
import model
import numpy as np
from rdkit import Chem
from collections import defaultdict
from Bio.Seq import MutableSeq
from Bio import SeqIO
import pandas as pd
import argparse

# Determine the absolute path of the script for loading pickle files
script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
print(script_path)

# Load necessary dictionaries
fingerprint_dict = model.load_pickle(os.path.join(script_path,'../../Data/input/fingerprint_dict.pickle'))
atom_dict = model.load_pickle(os.path.join(script_path,'../../Data/input/atom_dict.pickle'))
bond_dict = model.load_pickle(os.path.join(script_path,'../../Data/input/bond_dict.pickle'))
edge_dict = model.load_pickle(os.path.join(script_path,'../../Data/input/edge_dict.pickle'))
word_dict = model.load_pickle(os.path.join(script_path,'../../Data/input/sequence_dict.pickle'))

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    # words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]

    words = list()
    for i in range(len(sequence)-ngram+1) :
        try :
            words.append(word_dict[sequence[i:i+ngram]])
        except :
            word_dict[sequence[i:i+ngram]] = 0
            words.append(word_dict[sequence[i:i+ngram]])

    return np.array(words)
    # return word_dict

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    # atoms = list()
    # for a in atoms :
    #     try: 
    #         atoms.append(atom_dict[a])
    #     except :
    #         atom_dict[a] = 0
    #         atoms.append(atom_dict[a])

    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                # fingerprints.append(fingerprint_dict[fingerprint])
                # fingerprints.append(fingerprint_dict.get(fingerprint))
                try :
                    fingerprints.append(fingerprint_dict[fingerprint])
                except :
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    # edge = edge_dict[(both_side, edge)]
                    # edge = edge_dict.get((both_side, edge))
                    try :
                        edge = edge_dict[(both_side, edge)]
                    except :
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        predicted_value = self.model.forward(data)

        return predicted_value

# Define the function to parse mutants
def parse_mutant(wt_sequence_str, mutant_str):
    mutable_seq = MutableSeq(wt_sequence_str)

    for mut in mutant_str.split('_'):
        if mut[0].isdigit():
            position = int(mut[0:-1])
        else:
            position = int(mut[1:-1])
        new_residue = mut[-1]

        mutable_seq[position - 1] = new_residue

    return str(mutable_seq)

def main():
    parser = argparse.ArgumentParser(description='Perform Kcat value prediction for mutants.')
    parser.add_argument('-f','--fasta', type=str, required=True, help='Input FASTA file containing WT protein sequence.')
    parser.add_argument('-s','--substrate', type=str, required=True, help='Input substrate in SMILES format.')
    parser.add_argument('-m','--mutant_table', type=str, help='Input mutant table in text or CSV format.')
    parser.add_argument('--mutant_column', type=str, default='best_leaf', help='Name of the mutant column in the CSV file (default: best_leaf).')
    parser.add_argument('-o', '--output', type=str, default='./output/output.tsv', help='Output path and filename for saving prediction results (default: ./output/output.tsv).')
    args = parser.parse_args()

    fasta_file = args.fasta
    substrate_smiles = args.substrate
    mutant_table = args.mutant_table
    mutant_column = args.mutant_column


    
    
    # Output filename and path
    output_file = os.path.abspath(args.output)

    # Extract the output directory path from the provided --output argument
    output_dir = os.path.dirname(output_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

   

    if mutant_table: 
        # Parse WT protein sequence from FASTA file
        with open(fasta_file, 'r') as fasta_handle:
            for record in SeqIO.parse(fasta_handle, 'fasta'):
                wt_id=record.id
                
                wt_sequence = record.seq
                break  # Read only the first record (WT sequence)

        # Load mutant table using pandas
        if mutant_table.lower().endswith('.csv'):
            df_mutant_table = pd.read_csv(mutant_table)
            assert mutant_column in df_mutant_table.columns, f'Invalid mutation table column {mutant_column}. You must provide a column name from the input "--mutant_column"'
        else:
            df_mutant_table = pd.read_csv(mutant_table, sep='\t',names=[mutant_column])

        # Get the mutant names from the specified column
        
        mutant_names = df_mutant_table[mutant_column].tolist()

        # Compose the mutant dictionary
        mutants = {wt_id: wt_sequence}
        mutants.update({mutant_name:parse_mutant(wt_sequence_str=wt_sequence,mutant_str=mutant_name) for mutant_name in mutant_names})
    else:


        # Compose the full mutant dictionary from the fasta file if the mutation table is not specified.
        mutants = {record.id:str(record.seq) for record in SeqIO.parse( open(fasta_file, 'r'), 'fasta')}



    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    radius = 2
    ngram = 3
    dim = 10
    layer_gnn = 3
    side = 5
    window = 11
    layer_cnn = 3
    layer_output = 3
    lr = 1e-3
    lr_decay = 0.5
    decay_interval = 10
    weight_decay = 1e-6
    iteration = 100

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Transferred to GPU ...')
    else:
        device = torch.device('cpu')
    Kcat_model = model.KcatPrediction(device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(device)
    Kcat_model.load_state_dict(torch.load(os.path.join(script_path, '../../Results/output/all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50'), map_location=device))
    predictor = Predictor(Kcat_model)

    print('It\'s time to start the prediction!')
    print('-----------------------------------')

    with open(output_file, 'w') as outfile:
        items = ['Mutant Name', 'Substrate SMILES', 'Protein Sequence', 'Kcat value (1/s)']
        outfile.write('\t'.join(items) + '\n')

        for mutant_position, mutant_seq in mutants.items():
            # try:
            # Convert MutableSeq to string
            mutant_sequence = str(mutant_seq)

            if substrate_smiles and '.' not in substrate_smiles:
                mol = Chem.AddHs(Chem.MolFromSmiles(substrate_smiles))
                atoms = create_atoms(mol)
                i_jbond_dict = create_ijbonddict(mol)
                fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
                adjacency = create_adjacency(mol)
                words = split_sequence(mutant_sequence, ngram)

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                words = torch.LongTensor(words).to(device)

                inputs = [fingerprints, adjacency, words]

                print(f'Predicting molecule {substrate_smiles} against mutant {mutant_position}: {mutant_sequence}')

                prediction = predictor.predict(inputs)
                Kcat_log_value = prediction.item()
                Kcat_value = '%.4f' % math.pow(2, Kcat_log_value)

                line_item = [str(mutant_position), substrate_smiles, mutant_sequence, Kcat_value]
                outfile.write('\t'.join(line_item) + '\n')

            else:
                Kcat_value = 'None'
                substrate_smiles = 'None'
                print('Warning: No valid substrate SMILES found for mutant at position', mutant_position)
                line_item = [str(mutant_position), substrate_smiles, mutant_sequence, Kcat_value]
                outfile.write('\t'.join(line_item) + '\n')

            # except Exception as e:
            #     print(e)
            #     Kcat_value = 'None'
            #     line_item = [str(mutant_position), substrate_smiles, mutant_sequence, Kcat_value]
            #     outfile.write('\t'.join(line_item) + '\n')

    print('Prediction Done!')

if __name__ == '__main__':
    main()
