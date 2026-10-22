"""
This module implements the MATCHERS encoding of molecules, which is a representation that divides the connectivity 
of a molecule into atomic and bond features. 
The module provides an abstract base class `Matchers` and two concrete implementations: 
`OneHotMATCHERS` and `PeriodicTableMATCHERS`.
"""

import os
from abc import ABC, abstractmethod

from typing import Dict, List, Tuple, Union
import numpy as np

import networkx as nx
from . import atom_segmentation as atom_seg
from rdkit import Chem
from . import utils
import json

def min_max_scaling(value: int, inverted=False):
    """
    Min max scaling of the bond order to a range between 0 and 1. The bond order is expected to be in the range [0, 3].
    0 = no bond, 1 = single bond, 2 = double bond, 3 = triple bond. 
    The scaling is done to make the bond order compatible with the neural network input.
    """
    min_val = 0
    max_val = 3

    if not inverted: 
        return (value - min_val) / (max_val - min_val)
    # If inverted is True, the scaling is reversed to get the original bond order from the scaled value.
    elif inverted:  
        return value * (max_val - min_val) + min_val

class Matchers(ABC):
    """ 
    Creates the MATCHERS encoding of a molecule, where the connectivity is divided into atomic and bond features.
    """

    filler_value = 0
    atom_to_encoding=None

    def __init__(self, smiles=None, bond_mode=None, dict_saving_path=None) -> None:
        self.smiles = smiles
        if self.smiles is not None and not isinstance(smiles, str):
            raise TypeError("Argument 'smiles' should be a string")
        
        if self.smiles is not None and not self.is_supported_chemical():
            raise ValueError("This chemical is not supported.")
        
        if bond_mode is not None and bond_mode not in ['categorical', 'chemical']:
            raise ValueError("bond_mode must be 'categorical' or 'chemical'")
        self.bond_mode = bond_mode        

        if self.smiles is not None:
            self.matchers = self.to_matchers(self.smiles)

        # Length of one molecule plus the number of explicit H and formal charge
        self.atom_encoding_size = len(self.atom_to_encoding["C"]) + 2 
        
        try:
            with open(os.path.join(dict_saving_path, 'max_flags.json')) as f:
                self.flags_dict = json.load(f)
        except:
            self.flags_dict = None
        
    def is_supported_chemical(self) -> bool:
        return atom_seg.is_supported_chemical(self.smiles, verbosity="raise")

    def encode_atoms(self, graph) -> np.array:
        """
        Encodes the atoms of a molecule to an array.
        
        arguments: Graph representation of the molecule (includes graph of nodes, edges and type of edge, and nodes attributes)
        return: array with one-hot encoded data, NH's and formal charge. 
        """
        atom_encodings = list()

        for element_ix in range(graph.number_of_nodes()):
            element_data = graph.nodes[element_ix]
            n_explicit_h = element_data["n_explicit_h"]
            formal_charge = element_data["formal_charge"]
            atom_symbol = element_data["symbol"]
            atom_vec = self.atom_to_encoding[atom_symbol]

            atom_encoding = [*atom_vec, n_explicit_h, formal_charge]
            atom_encodings.append(atom_encoding)    # molecule is appended

        n_cols = len(atom_encodings[0])     # length of the molecule
        
        # Returns the atom features encoding with two rows for later usage, one for the isCyclic and the other isAcyclic flas
        return np.vstack(
            [
                np.array(atom_encodings),
                self.filler_value + np.zeros(n_cols),
                self.filler_value + np.zeros(n_cols),
            ]
        )
    
    def get_dictionary(self):
        return self.atom_to_encoding

    def to_matchers(self, smiles):
        """ 
        Creates the MATCHERS encoding of the a SMILES molecules by first transforming into a molecular object
        and then into a graph.
        arguments: None, SMILES already as an input for the class.
        returns: Array with atom and connectivity encoding. 
        """

        mol = Chem.MolFromSmiles(smiles)
        graph = utils.mol_to_graph(mol, bond_mode=self.bond_mode)
        atom_encodings = self.encode_atoms(graph)
        chains, paths = utils.apply_chain_decomposition(graph)
        connectivity = self.encode_connectivity(
            chains, paths, n_atoms=graph.number_of_nodes()
        )
        return np.hstack([atom_encodings, connectivity])

    def encode_connectivity(
        self,
        chains: List[List[Tuple[int, int, Dict[str, float]]]],
        paths: List[List[Tuple[int, int, Dict[str, float]]]],
        n_atoms: int,
    ) -> np.array:
        ''' Creates a connectivity matrix for the cycle and paths
        arguments: Chains and paths list containt the nodes and the information about the bond
        returns: Connectivity array '''

        n_subgraphs = len(chains) + len(paths)  # number of subgraphs
        # + 2 extra rows: one flag row for "is cyclic" (index -2), one flag row for "is acyclic" (index -1)
        connectivity = self.filler_value + np.zeros((n_atoms + 2, n_subgraphs))

        # Filling matrix with connectivity relationships:
        for chain_ix, chain in enumerate(chains):   # Chains by chains
            for edge in chain:      # Edges within a chain
                source, target, edge_data = edge    # Tuple: Node, Node, Bond Information
                bond_order = edge_data["bond_order"]  # For de novo: Take out.    
                if self.bond_mode == 'chemical':
                    bond_order = min_max_scaling(bond_order)

                connectivity[source, chain_ix] = bond_order     # Bond type at specific position

            is_cycle = chain[0][0] == chain[-1][1] and len(chain) > 2   # Checks if it is a cycle
            if is_cycle:
                connectivity[-2, chain_ix] = 1      # Ring flag

            if len(chain) == 1 or not is_cycle:
                connectivity[target, chain_ix] = bond_order     # Writes connectivity for the next step in cycle

        # Same as before, but for paths
        for path_ix, path in enumerate(paths, start=len(chains)):
            for edge in path:
                source, target, edge_data = edge
                bond_order = edge_data["bond_order"] # For de novo: Take out.
                if self.bond_mode == 'chemical':
                    bond_order = min_max_scaling(bond_order)
            
                connectivity[source, path_ix] = bond_order
                connectivity[target, path_ix] = bond_order
                connectivity[-1, path_ix] = 1  

        return connectivity

    def decode_connectivity(
        self,
        connectivity: np.array,
    ) -> List[Tuple[int, int, Dict[str, float]]]:
        
        ''' Decodes the connectivity of the graph into chains and bridges with Tuple informatio (vertex, vertex, bond info)
         argument: Connectivity matrix
          returns: List of chains '''

        connectivity = connectivity.T.tolist()  # n_chains x n_elements + 2
        connectivity = [subgraph for subgraph in connectivity if any(x != 0.0 for x in subgraph[:-2])]
        edges = list()
        for subgraph in connectivity:
            is_path = subgraph[-1]  # Chains of path
            if not is_path:     # for chains

                chain = utils.find_nonvalue_indices(subgraph[:-2], self.filler_value)

                # Skip invalid or too-short chains
                if not chain or len(chain) < 2 : 
                    continue

                starter_edge = (chain[0], chain[-1])
                bond_data = {"bond_order":  subgraph[chain[0]]}
                edges.append((*starter_edge, bond_data))
                for iterator in range(1, len(chain) - 1):
                    source = chain[iterator + 1]
                    target = chain[iterator]
                    bond_data = {"bond_order": subgraph[source]}
                    edges.append((source, target, bond_data))   # creates edges in list

                is_cycle = subgraph[-2] == 1
                if is_cycle:
                    bond_data = {"bond_order": subgraph[chain[1]]}
                    edges.append((chain[1], chain[0], bond_data))
            else:
                path = utils.find_nonvalue_indices(subgraph[:-2], self.filler_value)
                edges_in_path = [
                    (source, target, {"bond_order": subgraph[source]})
                    for source, target in zip(path[:-1], path[1:])
                ]

                edges.extend(edges_in_path)
        return edges


    def decode_atoms(self, atom_encodings: np.array) -> Dict[str, Union[int, float]]:
        n_atoms = atom_encodings.shape[0]
        atom_attributes = dict()
        for atom_ix in range(n_atoms):
            atom_encoding = atom_encodings[atom_ix, :]
            atom_symbol = self.atom_encoding_to_symbol(atom_encoding[:-2])
            n_explicit_h = atom_encoding[-2]
            formal_charge = atom_encoding[-1]
            atom_data = {
                "symbol": atom_symbol,
                "n_explicit_h": int(n_explicit_h),
                "formal_charge": int(formal_charge),
            }
            atom_attributes[atom_ix] = atom_data

        return atom_attributes

    def get_atom_encodings(self, encoded_atom):
        encoded_atom = np.array(encoded_atom)
        return encoded_atom[:-2, : self.atom_encoding_size]
        

    def get_connectivity(self, encoded_molecule):
        encoded_molecule = np.array(encoded_molecule)
        return encoded_molecule[:, self.atom_encoding_size :]

    @abstractmethod
    def atom_encoding_to_symbol(self, atom_encoding: np.array) -> str:
        pass

    def to_smiles(self, encoded_molecule) -> str:
        atom_encodings = self.get_atom_encodings(encoded_molecule)
        connectivity = self.get_connectivity(encoded_molecule)
        element_attributes = self.decode_atoms(atom_encodings)
        edges = self.decode_connectivity(connectivity)
        try:
            graph = nx.Graph(edges) 
            nx.set_node_attributes(graph, element_attributes)
            mol = utils.graph_to_mol(graph)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return edges
    
    @abstractmethod
    def __str__(self) -> str:
        pass


# Concrete implementations of MATCHERS, each defining a specific atom-encoding
# scheme (one-hot, or periodic-table features). Each atom_to_encoding dictionary is built
# once at class-definition time and is not expected to change afterwards.

class OneHotMATCHERS(Matchers):
    atom_to_encoding = {
            element: np.eye(utils.get_n_supported_elements()).tolist()[
                utils.get_index_of_element(element)
            ]
            for element in utils.get_supported_elements()
        }
    
    def atom_encoding_to_symbol(self, atom_encoding: np.array) -> str:
        hot_idx = atom_encoding.tolist().index(1)
        return utils.get_supported_elements()[hot_idx]

    def __str__(self) -> str:
        return "OneHot - MATCHERS"

class PeriodicTableMATCHERS(Matchers):
    atom_to_encoding = {
            element_symbol: list(atom_seg.st_pt3_features[element_symbol].values())
            for element_symbol in utils.get_supported_elements()
        }
    
    atomic_mass_to_sym = {
        features["mass"]: symbol for symbol, features in atom_seg.st_pt_features.items()
    }

    def atom_encoding_to_symbol(self, atom_encoding: np.array) -> str:
        atomic_mass = atom_encoding[0]
        return self.atomic_mass_to_sym[atomic_mass]

    def __str__(self) -> str:
        return "Periodic Table - MATCHERS"
