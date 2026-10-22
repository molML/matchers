# MATCHERS

MATCHERS is a matrix representation of molecules that separates **atom features** from **chemical topology**.
Each molecule (given as SMILES) is turned into a matrix with one row per atom:

- **Atom block:** the element encoding (one-hot or periodic-table features), followed by the number of
  explicit hydrogens and the formal charge.
- **Connectivity block:** one column per subgraph of the molecule (closed cyclic, open cyclis, and acyclic paths). Each entry
  holds the bond order between the atoms of that subgraph.
- **Two flag rows** at the bottom mark each subgraph column as a ring (`is ring`) or an acyclic path (`is path`).

The encoding can be decoded back to SMILES, which makes it usable for both property prediction and
de novo molecule generation.

## Environment

python (v.3.9.7), networkx (v.3.1), rdkit (v.2022.03.1), regex (v.2025.9.1), pandas (v.2.0.0), numpy (v.1.21.3)

## Installation

```bash
git clone <repository-url>
cd <repository-folder>
conda create -n matchers python=3.9.7
conda activate matchers
conda install -c conda-forge rdkit=2022.03.1
pip install numpy==1.21.3 networkx==3.1 regex==2025.9.1
```

Optional, depending on the model you train with: `torch` or `tensorflow`.
MATCHERS itself only returns NumPy arrays and needs neither.

## Representations

| Class | Atom encoding |
| `OneHotMATCHERS` | One-hot vector over the supported elements |
| `PeriodicTableMATCHERS` | Min-max scaled periodic-table features (mass, van der Waals volume, electronegativity) |

**Supported chemistry:** C, H, O, N, S, P, F, Cl, Br, I (including aromatic c, n, o, s); at most 50 rings.
SMILES strings need to be canonicalized before with RDKit.
Not supported: isotopes / atomic masses in brackets, uncharged single-atom metals.
Stereochemistry is **not** encoded.

## Usage

### Encode and decode a single molecule

```python
from matchers.matchers.matchers import OneHotMATCHERS

encoder = OneHotMATCHERS("CC(=O)Oc1ccccc1C(=O)O", bond_mode="categorical")
matrix = encoder.matchers                 # numpy array: (n_atoms + 2, atom features + subgraphs)
smiles = encoder.to_smiles(matrix)        # canonical SMILES, without stereochemistry
```

`bond_mode`:
- `"categorical"`: bonds encoded as 1 = single, 2 = aromatic, 3 = double, 4 = triple. **Required for decoding.**
- `"chemical"`: bonds encoded as min-max scaled bond orders (1, 1.5, 2, 3). Encoding only.

### Encode a dataset (vectorizer)

The vectorizer brings all molecules to the same shape, so they can be used as model input:

1. `fit()` determines the maximum number of closed cyclic paths ('closed chains'), open cyclic paths ('open chains'), and acyclic paths ('paths') in the corpus
   (stored in `max_flags.json`) and saves the atom vocabulary (`segment2vec.json`).
2. `vectorize_chemicals()` sorts each molecule's subgraph columns into these three blocks and pads every block
   to its maximum. It drops the two flag rows and pads the rows (atoms) to `maxlen` with the padding value
   (`-999`). Padding value can be changed, depending on the task.
3. If performing de novo design, adding a start and end character is necessar

```python
from matchers.vectorizer.matchers_vectorizers import OneHotMatchersVectorizer

vectorizer = OneHotMatchersVectorizer(
    maxlen=100,                  # max. number of atoms (+2 if add_start_end=True)
    bond_mode="categorical",
    add_start_end=False,         # True: adds START/END rows for autoregressive generation
    saving_dir="output/",        # max_flags.json and segment2vec.json are written here
)
vectorizer.fit(train_smiles)
X = vectorizer.vectorize_chemicals(train_smiles)   # numpy, shape (n_molecules, maxlen, features)
```

The output is a NumPy array. Convert it for your framework:

```python
import torch
X_torch = torch.from_numpy(X).float()   # PyTorch
# Keras / TensorFlow accept the NumPy array directly.
```

**Note:** if `saving_dir` already contains a `max_flags.json`, `fit()` reuses it instead of refitting.
Use a fresh folder when you change the dataset.

### Decode vectorized molecules

The flag rows are removed during vectorization. To decode, remove the padding and the START/END rows,
reintroduce the flags, then decode:

```python
import numpy as np
from matchers.vectorizer import utils as vec_utils

representation = vectorizer.representation()
for padded in X:
    real = padded[~np.all(padded == vectorizer.padding_value, axis=1)]   # remove row padding
    if vectorizer.add_start_end:
        real = real[1:-1]                                                # remove START/END rows
    full = vec_utils.reintroduce_flags(
        real, vectorizer.atom_encoding_size, vectorizer.standardization_dict
    )
    smiles = representation.to_smiles(full)
```

## Package structure

```
matchers/
├── __init__.py
├── test_roundtrip.py              # encode/decode round-trip tests
├── chemistry/
│   └── chemistry.py               # periodic-table properties, element/aromaticity checks
├── data/
│   └── periodic_table_properties.json
├── matchers/
│   ├── matchers.py                # MATCHERS base class, OneHotMATCHERS, PeriodicTableMATCHERS
│   ├── atom_segmentation.py       # supported-chemistry check, scaled periodic-table features
│   └── utils.py                   # molecule to graph conversion, chain decomposition by Schmidt et al., creating sequential paths
├── sequence/
│   └── smiles_processing.py       # SMILES cleaning, tokenization, validation
├── utils/
│   └── utils.py                   # JSON helpers, package path
└── vectorizer/
    ├── matchers_vectorizers.py    # batch encoding, standardization (ordering of the ) and padding
    └── utils.py                   # padding, column reordering, flag reintroduction
```

## Round-trip test

`test_roundtrip.py` checks that molecules come back unchanged after encoding and decoding, on a random sample
of a SMILES file (one SMILES per line):

| Test | Path |
|---|---|
| 1 | Encode, then decode directly with the MATCHERS class |
| 2 | Encode, vectorize (standardize and pad), reintroduce the flags, then decode |
| 3 | Same as test 2, with START/END vectors |

Input SMILES are converted to RDKit canonical form (without stereochemistry) first, so input and output can be
compared as strings. The sample size, seed, representation and tests are set at the top of `main()`.

Run it from the folder **above** the package (otherwise `import matchers` resolves to the inner
`matchers/matchers/` folder):

```bash
cd <folder containing matchers/>
python -m matchers.test_roundtrip
```

## Known limitations

- Only `bond_mode="categorical"` can be decoded.
- If SMILES are not canonicalized with RDKit, then SMILES before encoding and after decoding will not be the same.
- `vectorize_chemicals` raises an error if a molecule has more atoms than `maxlen`. Molecules are never
  truncated.
- If none of the fitted molecules has a subgraph of one type (closed chain, open chain or path),
  `vectorize_chemicals` fails with a "negative dimensions" error. Fit on a large, diverse set of molecules.
- `add_start_end=True` extends `OneHotMATCHERS`' atom vocabulary at class level, for the rest of the Python
  process.
- MATCHERS cannot encode carbenes.

## License

MIT License