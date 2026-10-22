import networkx as nx
from rdkit import Chem

__SUPPORTED_ELEMENTS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]
__ELEMENT_TO_IDX = {element: idx for idx, element in enumerate(__SUPPORTED_ELEMENTS)}

__BOND_ORDER_TO_RDKIT_BOND_CATEGORICAL = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.AROMATIC,
    3: Chem.rdchem.BondType.DOUBLE,
    4: Chem.rdchem.BondType.TRIPLE,
}

__RDKIT_BOND_TO_BOND_ORDER_CHEMICAL = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.AROMATIC: 1.5,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
}

__RDKIT_BOND_TO_BOND_ORDER_CATEGORICAL = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.AROMATIC: 2,
    Chem.rdchem.BondType.DOUBLE: 3,
    Chem.rdchem.BondType.TRIPLE: 4,
}

def get_supported_elements():
    return __SUPPORTED_ELEMENTS

def get_n_supported_elements():
    return len(__SUPPORTED_ELEMENTS)

def get_index_of_element(element):
    return __ELEMENT_TO_IDX.get(element)

def mol_to_graph(mol: Chem.Mol, bond_mode: str = 'categorical'):
    """
    Transforms the molecule to a graph with the features Symbol, NH's and Formal charge.
    arguments: mol (smiles need to be transform into a mol)
    returns: graph with node attribute
    """

    # Select the appropriate bond dictionary based on the bond_mode argument
    # For 'categorical', bond orders are represented as integers (1, 2, 3, 4). Used for de novo.
    # For 'chemical', bond orders are represented as floats (1.0, 1.5, 2.0, 3.0). Used for not de novo.
    if bond_mode == 'categorical':
        bond_dict = __RDKIT_BOND_TO_BOND_ORDER_CATEGORICAL
    elif bond_mode == 'chemical':
        bond_dict = __RDKIT_BOND_TO_BOND_ORDER_CHEMICAL

    # Create a list of bonds and a dictionary of elements with their features.
    bond_list = list()
    elements = dict()
    for atom in mol.GetAtoms():    # Sequence containing all atoms, parsed via canonical ordering
        atom_ix = atom.GetIdx()     # Order in molecule (0, 1, 2, 3, ...)
        # Extract the features of the atom and store them in a dictionary.
        element_features = {
            "symbol": atom.GetSymbol(),
            "n_explicit_h": atom.GetNumExplicitHs(),
            "formal_charge": atom.GetFormalCharge(),
        }
        elements[atom_ix] = element_features
        # Iterate over the bonds of the atom to create a list of bonds with their features.
        for bond in atom.GetBonds():
            source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()   # Position of the bonds within two atoms
            neighbor_idx = target
            if target == atom_ix:   # bond may be stored from either endpoint; pick the atom that isn't atom_ix
                neighbor_idx = source

            if atom_ix < neighbor_idx:      # needs to start at the lowest one
                rdkit_bond = bond.GetBondType()
                bond_order = bond_dict.get(rdkit_bond, 0.0)  # defaults to 0.0 if the bond type isn't in the mapping
                bond_list.append((atom_ix, neighbor_idx, {"bond_order": bond_order}))   # creates a list of tuples where atom types are
    
    g = nx.Graph(bond_list)     # creation of the graph
    nx.set_node_attributes(g, elements)     # set to the node (vertices) the elements attributed in the dicitonary above
    return g

def decompose_graph_into_chains_and_bridges(graph):
    """
    Creates lists for chains (cycle) and paths. Each list consists of tuples in which the edge 
    (chain and bridges) and the attributes dictionary of the edge available.
    arguments: graph with node features included
    returns: list of chains and bridges with attribute dicitonary to the edge
    """
    
    chains = list(nx.chain_decomposition(graph))    # list of chains (see online example, only ones belonging to a cycle???)
    chain_edges = set().union(*[set(chain) for chain in chains])    # remove duplicate edges

    bridges = list()
    for edge in graph.edges:
        rev_edge = (edge[1], edge[0])
        # If edge not in chains, then bridge
        if edge not in chain_edges and rev_edge not in chain_edges:
            if edge[0] < edge[1]:
                bridges.append(edge)
            else:
                bridges.append(rev_edge)

    # list of each edge with attribute dictionary to edge (type of bond; for both chains and bridges)
    chains_with_bonds = [
       [(*edge, graph.get_edge_data(*edge)) for edge in chain] for chain in chains
    ]
    bridges_with_bonds = [(*edge, graph.get_edge_data(*edge)) for edge in bridges]
    return chains_with_bonds, bridges_with_bonds


def apply_chain_decomposition(graph):
    ''' 
    Chain decomposition by compressing the bridges together to a path.
    arguments: Graph representation of a molecule.
    returns: chains and paths '''
    chains, bridges = decompose_graph_into_chains_and_bridges(graph)

    bridge_graph = nx.Graph(bridges)    # Graph of the bridges is created
    # subgraphs bridges are created, (max length must be 2)
    components = [
        bridge_graph.subgraph(component).copy()
        for component in nx.connected_components(bridge_graph)  # which bridges are connected 
    ]

    paths = list()
    for component in components:
        traversal = list(nx.dfs_edges(component))   # Iterates over the edges of the connected bridges
        visited_nodes = set()
        path = list()
        last_target = -1
        for source, target in traversal:   # Iteration through every single bridge (source, target)
            if source > target:
                source, target = target, source     # change order from smaller to higher

            # all edges in a path visited
            if source in visited_nodes or (source != last_target and last_target != -1):
                paths.append(path)      # close off the current path
                path = list()            # start a new path
                visited_nodes = set()   # reset visited-node tracking for the new path
                last_target = -1

            last_target = target
            path.append((source, target, component.get_edge_data(source, target)))  # add of the bridge edge
            visited_nodes.add(source)

        if len(path) > 0:
            paths.append(path)
    return chains, paths

def graph_to_mol(graph):
    """
    Recreates the Molecule from the Graph stepwise through addition of Atoms and Bonds.
    Arguments: Graph respresentation of molecule
    Returns: Molecular Object 
    """
    mol = Chem.RWMol()
    nodes = graph.nodes.data()
    edges = graph.edges.data()
    node_graph_ix_to_mol_ix = dict()
    for node_graph_ix, node_data in nodes:
        atom = Chem.Atom(node_data["symbol"])
        atom.SetNumExplicitHs(node_data["n_explicit_h"])
        atom.SetFormalCharge(node_data["formal_charge"])
        node_mol_ix = mol.AddAtom(atom)
        node_graph_ix_to_mol_ix[node_graph_ix] = node_mol_ix

    for edge in edges:
        source, target, edge_data = edge
        bond_order = edge_data["bond_order"]
        # Convert the bond order to the corresponding RDKit bond type using the mapping dictionary.
        # Here, we use the categorical mapping to ensure that the bond type 
        # is correctly interpreted by RDKit. As it is normally used only in 
        rdkit_bond = __BOND_ORDER_TO_RDKIT_BOND_CATEGORICAL[bond_order]
        mol.AddBond(
            node_graph_ix_to_mol_ix[source], node_graph_ix_to_mol_ix[target], rdkit_bond
        )

    return mol.GetMol()

def find_nonvalue_indices(lst, value):
    return [i for i, elem in enumerate(lst) if elem != value]
