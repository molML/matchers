from abc import ABC, abstractmethod
from typing import List
import os
import numpy as np
from matchers.matchers.matchers import OneHotMATCHERS, PeriodicTableMATCHERS
from . import utils
import json

class MatchersVectorizer(ABC):
    representation = None
    padding_value = None

    def __init__(self, maxlen: int, bond_mode: str = 'categorical', add_start_end: bool = False, given_padding_value = None, saving_dir: dict = None) -> None:
        if maxlen < 3:
            raise ValueError("Sequence length must be larger than 3.")
        
        self.maxlen = maxlen
        self.maxheight = -1

        # For the autoregressive generation case, we need to add start and end tokens to the sequences.
        self.add_start_end = add_start_end
        if self.add_start_end:
            # Handles the case for one-hot encoding, where we need to expand the vocabulary 
            # to include START and END tokens.
            self._expand_vocabulary()

        self.atom_encoding_size = getattr(self.representation(), 'atom_encoding_size')

        self.saving_dir = saving_dir
        self.bond_mode = bond_mode

        self.padding_value = given_padding_value if given_padding_value is not None else self.__class__.padding_value

    def fit(self, smiles_corpus: List[str], shuffle: bool = False):

        # "Standardization" dictionary is used to store the maximum number of different types of paths
        # This is important for ensuring that all molecules are represented with the same number of features (especially as flags are eliminated).
        standardization_path = os.path.join(self.saving_dir, 'max_flags.json')

        try:   # Load the standardization dictionary if it exists, if standardization already took place
            with open(standardization_path, 'r') as json_file:
                self.standardization_dict = json.load(json_file)

        except (FileNotFoundError, json.JSONDecodeError):
            print('No dictionary found for standardization. Fitting the dictionary!')
            matchers_batch = [
                self.representation(smiles, self.bond_mode).matchers for smiles in smiles_corpus
                ]

            # Calculate the maximum number of different types of paths for each molecule in the batch
            self.standardization_dict = utils.calculate_maxlen_flags(
                matchers_batch, 
                self.atom_encoding_size)

            with open(standardization_path, 'w') as json_file:
                json.dump(self.standardization_dict, json_file)

        num_features = sum(self.standardization_dict.values())
        self.maxheight = num_features + self.atom_encoding_size

        segment2vec = self.representation().get_dictionary()

        if shuffle:
            # Shuffle the segment2vec dictionary
            keys = list(segment2vec.keys())
            values = list(segment2vec.values())
            np.random.shuffle(values)
            segment2vec = dict(zip(keys, values))

            # Overwrite the class attribute so the shuffled version is used
            self.representation.atom_to_encoding = segment2vec

        with open(os.path.join(self.saving_dir, 'segment2vec.json'), 'w') as json_file:
            json.dump(segment2vec, json_file)

    def vectorize_chemicals(self, smiles_corpus: List[str]):

        if self.maxheight == -1:
            raise ValueError(
                "Fit the vectorizer before use or provide maxheight (atom encoding dimension and chemical topology matrix.)"
            )

        if not self.standardization_dict:
            raise ValueError(
                "Fit the vectorizer before use or provide the maximum number of subchannels."
            )

        matchers_batch = [self.representation(smiles, self.bond_mode).matchers for smiles in smiles_corpus]

        # Reorder the matrices based on the standardization dictionary and pad each category to the maximum length
        hori_padded_matchers = utils.reorder_matrices_by_flags(
            matchers_batch,
            self.atom_encoding_size,
            self.standardization_dict,)

        # Dropping the last two rows of the padded matchers, which correspond to the 
        # flags (as they are not needed for the actual generation of molecules, but only for standardization purposes).
        hori_padded_matchers = [arr[:-2, :] for arr in hori_padded_matchers]

        # If necessary, adding the start and end vectors to the sequences.
        # (Important for autoregressive generation tasks)
        if self.add_start_end:

            # Calculating the total length of the vector (atom encoding + connection encoding)
            total_vector_length = hori_padded_matchers[0].shape[1]

            # Defining the start and end vectors based on the atom encoding size and connection encoding size
            atom_len = self.atom_encoding_size
            conn_length = total_vector_length - atom_len

            # Defining the start and end vectors.
            # The start and end vectors for the atom encoding is depended on the type of atom encoding (periodic table properties
            # or one hot encoding) and fives for the connection encoding.
            start_vector = np.concatenate([
                self._get_start_atom_vector(), np.ones(conn_length) * 5
            ])[np.newaxis, :]
            end_vector = np.concatenate([
                self._get_end_atom_vector(), np.zeros(conn_length)
            ])[np.newaxis, :]

            # Add start and end vectors to each sequence
            hori_padded_matchers = utils.add_start_end_character(
                hori_padded_matchers,
                start_vector=start_vector,
                end_vector=end_vector,
            )

        matchers_padded = utils.pad_chemical_sequences(
            chemical_sequences=hori_padded_matchers,
            maxlen=self.maxlen,
            padding_value=self.padding_value,
        )
        
        # if self.bond_mode == 'categorical':
        #     # Used for de novo generation (torch model) where the bond type is represented as a categorical variable.
        #     matchers_padded = utils.pad_sequences_torch(
        #         hori_padded_matchers,
        #         maxlen=self.maxlen,
        #         padding_value=self.padding_value,
        #     )
        # else:
        #     # Pad the sequences to the maximum length
        #     matchers_padded = utils.pad_sequences_keras(
        #         hori_padded_matchers,
        #         maxlen=self.maxlen,
        #         padding_value=self.padding_value,
        #     )

        return matchers_padded


    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def _expand_vocabulary(self):
        pass

    @abstractmethod
    def _get_start_atom_vector(self):
        pass

    @abstractmethod
    def _get_end_atom_vector(self):
        pass


class OneHotMatchersVectorizer(MatchersVectorizer):
    # Calling the representation class directly to access its attributes
    representation = OneHotMATCHERS
    padding_value = -999

    def __str__(self) -> str:
        return "OneHot - MATCHERS"

    def _expand_vocabulary(self):
        # Guard: only expand once. Good for HP loop iteration.
        # Since atom_to_encoding is a class-level attribute,
        # it persists across encoder instances (e.g. across HP loop iterations).
        if "START" in self.representation.atom_to_encoding:
            return

        # Expand the vocabulary to include START and END tokens
        # Current number of unique entries in the atom_to_encoding dictionary
        n = len(self.representation.atom_to_encoding["C"])

        # Pad all existing entries with 2 trailing zeros (one for START and one for END)
        for key in self.representation.atom_to_encoding:
            self.representation.atom_to_encoding[key] = self.representation.atom_to_encoding[key] + [0.0, 0.0]

        # Add START and END as new entries
        start_vec = [0.0] * n + [1.0, 0.0]  # position n   = START
        end_vec   = [0.0] * n + [0.0, 1.0]  # position n+1 = END

        self.representation.atom_to_encoding["START"] = start_vec
        self.representation.atom_to_encoding["END"] = end_vec

    def _get_start_atom_vector(self):
        start_atom = np.array(self.representation.atom_to_encoding["START"])
        return np.concatenate([start_atom, [0.0, 0.0]]) # for n explicit H and formal charge

    def _get_end_atom_vector(self):
        end_atom = np.array(self.representation.atom_to_encoding["END"])
        return np.concatenate([end_atom, [0.0, 0.0]])   # for n explicit H and formal charge


class PeriodicTableMatchersVectorizer(MatchersVectorizer):
    representation = PeriodicTableMATCHERS
    padding_value = -999

    def __str__(self) -> str:
        return "Periodic Table - MATCHERS"

    def _expand_vocabulary(self):
        pass

    def _get_start_atom_vector(self):
        return np.ones(self.atom_encoding_size)

    def _get_end_atom_vector(self):
        return np.zeros(self.atom_encoding_size)
