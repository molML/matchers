import numpy as np

def pad_chemical_sequences(chemical_sequences, maxlen, padding_value):
    if not chemical_sequences:
        raise ValueError("The input chemical_sequences list is empty.")

    chemical_sequences = [np.array(seq) for seq in chemical_sequences]
    feature_size = chemical_sequences[0].shape[1]

    if maxlen is None:
        raise ValueError("Maximum length of MATCHERS must be specified beforehand.")

    batch_size = len(chemical_sequences)
    # Creates a 3D array filled with the padding value
    padded_array = np.full((batch_size, maxlen, feature_size), padding_value, dtype=float)

    for i, seq in enumerate(chemical_sequences):
        seq_len = seq.shape[0]
        padded_array [i, :seq_len, :] = seq

    return padded_array

def pad_sequences_torch(sequences, maxlen=None, padding_value=0.0, dtype=None, device=None):
    """
    Torch wrapper of pad_chemical_sequences. Imports torch lazily so that importing
    this module doesn't require torch to be installed unless this specific function
    is actually called (nothing else in this file needs it).
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    np_padded = pad_chemical_sequences(sequences, maxlen=maxlen, padding_value=padding_value)
    tensor = torch.tensor(np_padded, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def pad_sequences_keras(chemical_sequences, maxlen, padding_value):
    """
    Keras wrapper of pad_chemical_sequences. Imports tensorflow lazily so that importing
    this module doesn't require tensorflow to be installed unless this specific function.
    """
    import tensorflow as tf
    
    return tf.keras.preprocessing.sequence.pad_sequences(
        chemical_sequences,
        maxlen=maxlen,
        truncating="pre",
        padding="post",
        value=padding_value,
        dtype=float,
    )

def add_start_end_character(arrays, start_vector=None, end_vector=None):

    output = []

    for array in arrays:

        array = np.array(array)     # incase it is not a numpy array
        rows, cols = array.shape    # shape of the array

        if start_vector is None:
            raise ValueError("The start_vector must be provided.")
        else:
            assert start_vector.shape == (1, cols), \
                f"The start_vector must have shape (1, {cols}), got {start_vector.shape}"

        if end_vector is None:
            raise ValueError("The end_vector must be provided.")
        else:
            assert end_vector.shape == (1, cols), \
                f"The end_vector must have shape (1, {cols}), got {end_vector.shape}"


        padded = np.vstack([start_vector, array, end_vector])
        output.append(padded)

    return output

def calculate_maxlen_flags(encoded_batch, atom_encoding_size):
    """
    Calculate the maximum length of each type of path across all molecules in the batch.
    """
    max_closed_chains, max_open_chains, max_path = 0, 0, 0
    for ix, encoded_molecule in enumerate(encoded_batch):
        connectivity = encoded_molecule[:, atom_encoding_size:]

        num_closed_chains = np.sum(connectivity[-2, :])
        num_path = np.sum(connectivity[-1, :])
        num_open_chains = len(connectivity[-1]) - (num_closed_chains + num_path)

        if max_closed_chains < num_closed_chains:
            max_closed_chains = num_closed_chains

        if max_open_chains < num_open_chains:
            max_open_chains = num_open_chains

        if max_path < num_path:
            max_path = num_path

    return {'closed_chains': max_closed_chains, 'open_chains': max_open_chains, 'path': max_path}


def reorder_matrices_by_flags(encoded_batch, atom_encoding_size, max_size_dict):
    padded_molecules = list()

    for encoded_molecule in encoded_batch:

        # Splitting the encoded molecule into atom encoding and connectivity parts
        atom_encoding = encoded_molecule[:, : atom_encoding_size]
        connectivity = encoded_molecule[:, atom_encoding_size :]

        closed_chains, open_chains, path = [], [], []

        for subgraph in connectivity.T.tolist():
            if subgraph[-2]:
                closed_chains.append(subgraph)
            elif subgraph[-1]:
                path.append(subgraph)
            else:
                open_chains.append(subgraph)

        separated_subgraphs = {'closed_chains': closed_chains, 'open_chains': open_chains, 'path': path}
        padded_connectivity = atom_encoding
        for name, subgraph in separated_subgraphs.items():
            if not subgraph:
                sorted_subgraph = np.zeros((atom_encoding.shape[0], 1))
            else:
                subgraph = np.transpose(subgraph)
                sorted_indices = np.argsort(np.count_nonzero(subgraph[:-2, :], axis=0))[::-1]
                sorted_subgraph = subgraph[:, sorted_indices]

            n_pad_columns = int(max_size_dict[name]) - sorted_subgraph.shape[1]

            pad_columns = np.zeros((sorted_subgraph.shape[0], n_pad_columns))
            padded_subgraph = np.hstack((sorted_subgraph, pad_columns))

            padded_connectivity = np.hstack((padded_connectivity, padded_subgraph))

        padded_molecules.append(padded_connectivity)
    return padded_molecules

def eliminate_padding(padded_molecules, atom_encoding_size):

    atom_encoding = padded_molecules[:, : atom_encoding_size]
    connectivity = padded_molecules[:, atom_encoding_size :]

    non_zero_columns = np.where(~np.all(connectivity == 0, axis=0))[0]
    unpadded_connectivity = connectivity[:, non_zero_columns]
    unpadded_molecule = np.hstack((atom_encoding, unpadded_connectivity))

    return unpadded_molecule


def reintroduce_flags(molecule, atom_len, flags_dict):
    """
    Appends two flag rows to MATCHERS matrices. To reintroduce the flags, the function takes the original 
    molecule matrix (without flags), the atom encoding length, and a dictionary containing the counts of each type of path 
    (closed cyclic paths, open cyclic paths, and acyclicpaths). It then reconstructs the flags and appends them to the 
    connectivity part of the molecule matrix.
    Returns: 
    full_molecule_matrix
    """
    mol_array = np.array(molecule)  # shape: (n_atoms, n_features (atom_features + chemical topology))
    atom_part = mol_array[:, :atom_len]    # shape: (n_atoms, atom features dimension)
    _, atom_dim = atom_part.shape
    connectivity_part = mol_array[:, atom_len:]     # shape: (n_atoms, chemical topology dimension)  
    _, n_subgraphs = connectivity_part.shape

    # Rebuild the flags 
    n_closed = int(flags_dict.get('closed_chains', 0))
    n_open   = int(flags_dict.get('open_chains', 0))
    n_paths  = int(flags_dict.get('path', 0))

    # Closed rings
    is_cycle_flags = np.zeros((1, n_subgraphs), dtype=float)
    is_cycle_flags[0, :n_closed] = 1.0    
    # Paths   
    is_path_flags  = np.zeros((1, n_subgraphs), dtype=float)
    is_path_flags[0, n_closed + n_open : n_closed + n_open + n_paths] = 1.0

    # Create connectivity matrix with flags
    connectivity_with_flags = np.vstack([
        connectivity_part,
        is_cycle_flags,
        is_path_flags
    ])

    # Build corresponding atom flag rows (zero-rows)
    atom_flag_rows = np.zeros((2, atom_dim), dtype=float)
    atom_with_flags = np.vstack([
        atom_part,
        atom_flag_rows
    ])

    return np.hstack([
        atom_with_flags,
        connectivity_with_flags
    ])

