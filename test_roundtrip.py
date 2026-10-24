"""
Three MATCHERS round-trip tests on a random sample of a SMILES file. Following tests are performed:

Test 1:  Encode and decode directly with the MATCHERS class (flags still in the matrix, no vectorizer).
         Checked molecule by molecule.
Test 2:  Encode, then standardize/pad (vectorize_chemicals, flags dropped), un-pad, reintroduce the flags, and decode.
Test 3:  Same as test 2, but with START / END vectors added.

For every molecule that does not come back identical, the input SMILES and the decoded output are printed.
"Identical" means: equal to the RDKit canonical SMILES of the input WITHOUT stereochemistry, as MATCHERS does not
encode stereochemistry. The canonical SMILES is printed for reference.

Place this script NEXT TO the `matchers/` package folder (not inside it), otherwise `import matchers`
resolves to the inner `matchers/matchers/` folder.

Notes:
* Decoding is only necessary in cases of de novo drug generation. Tests 2 and 3 use bond_mode='categorical'
  (the only mode graph_to_mol can decode until now). vectorize_chemicals pads with numpy, so neither torch nor
  tensorflow is needed.
* Test 3 expands OneHotMATCHERS.atom_to_encoding at class level (START/END). The tests therefore always run in the
  order 1, 2, 3 and test 3 is the last thing that touches OneHotMATCHERS in the process. Use a new process
  (new seed) for another run.
"""

import os
import random
import tempfile

import numpy as np
from rdkit import Chem, RDLogger

from matchers.matchers.matchers import OneHotMATCHERS, PeriodicTableMATCHERS
from matchers.vectorizer.matchers_vectorizers import (
    OneHotMatchersVectorizer,
    PeriodicTableMatchersVectorizer,
)
from matchers.vectorizer import utils as vec_utils

RDLogger.DisableLog("rdApp.*")

REPRESENTATIONS = {
    "onehot": (OneHotMATCHERS, OneHotMatchersVectorizer),
    "periodic": (PeriodicTableMATCHERS, PeriodicTableMatchersVectorizer),
}

# Helpers: Loading of a .smiles file, and canonicalization of SMILES without stereochemistry
# (for comparison with decoded output)
def load_smiles(path, limit, seed):
    with open(path) as f:
        smiles = [line.strip() for line in f if line.strip()]
    if limit and limit < len(smiles):
        smiles = random.Random(seed).sample(smiles, limit)
    return smiles

def canonical_no_stereo(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)

# Reporting class
class Report:
    """Counts matches and prints every mismatch."""

    def __init__(self, title, n_total):
        self.title, self.n_total = title, n_total
        self.n_match = 0
        self.n_fail = 0

    def check(self, smiles, decoded):
        if decoded == smiles:
            self.n_match += 1
            return
        self.n_fail += 1
        print(f"[MISMATCH]\n   in : {smiles}\n   out: {decoded}")

    def error(self, smiles, exc):
        self.n_fail += 1
        print(f"[ERROR]\n   in : {smiles}\n   out: <{type(exc).__name__}: {exc}>")

    def finish(self):
        pct = 100 * self.n_match / self.n_total if self.n_total else 0.0
        print(f"--> {self.title}: identical {self.n_match}/{self.n_total} ({pct:.2f}%), failed {self.n_fail}")
        return self.n_fail == 0


# Test 1: Direct encode -> decode with MATCHERS class (no vectorizer used)
def test_direct(representation_cls, smiles_list):
    report = Report("TEST 1: MATCHERS encode -> decode (flags kept, no vectorizer)", len(smiles_list))
    for smiles in smiles_list:
        try:
            encoder = representation_cls(smiles, bond_mode="categorical")
            decoded = encoder.to_smiles(encoder.matchers)
        except Exception as exc:
            report.error(smiles, exc)
            continue
        report.check(smiles, decoded)
    return report.finish()


# Tests 2 and 3: through the vectorizer
def decode_vectorized(padded, vectorizer, representation):
    matrix = np.asarray(padded, dtype=float)                                   # (maxlen, maxheight)

    # 1) remove the row padding at the bottom
    real = matrix[~np.all(matrix == vectorizer.padding_value, axis=1)]

    # 2) remove START / END rows
    if vectorizer.add_start_end:
        real = real[1:-1]

    # 3) put the two path flag rows back (from the standardized column blocks)
    full = vec_utils.reintroduce_flags(
        real, vectorizer.atom_encoding_size, vectorizer.standardization_dict
    )

    # 4) decode
    return representation.to_smiles(full)


def test_vectorized(vectorizer_cls, smiles_list, add_start_end, batch_size):
    title = "TEST 3: encode -> standardize -> + START/END -> decode" if add_start_end \
        else "TEST 2: encode -> standardize -> reintroduce flags -> decode"
    report = Report(title, len(smiles_list))

    # padded length = largest molecule in the sample (+2 rows for START/END)
    max_atoms = max(Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles_list)
    maxlen = max_atoms + (2 if add_start_end else 0)

    with tempfile.TemporaryDirectory() as saving_dir:       # fresh dir -> no stale max_flags.json
        vectorizer = vectorizer_cls(
            maxlen=maxlen,
            bond_mode="categorical",
            add_start_end=add_start_end,
            saving_dir=saving_dir,
        )
        representation = vectorizer.representation()

        vectorizer.fit(smiles_list)
        print("Standardization dictionary:", vectorizer.standardization_dict, "| maxlen:", maxlen)

        for start in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[start:start + batch_size]
            batch = vectorizer.vectorize_chemicals(chunk)        # numpy, (len(chunk), maxlen, maxheight)
            for smiles, padded in zip(chunk, batch):
                try:
                    decoded = decode_vectorized(padded, vectorizer, representation)
                except Exception as exc:
                    report.error(smiles, exc)
                    continue
                report.check(smiles, decoded)

    return report.finish()


# Main
def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The script sits next to the `matchers/` folder; adjust if clean.smiles lives elsewhere.
    smiles_path = os.path.join(base_dir, "clean.smiles")

    # Some variables to change for testing purposes (e.g. to limit the number of molecules in the sample)
    limit = 60000
    representation = "onehot"  # possible: "onehot", "periodic", "both"
    seed = 65
    batch_size = 5
    tests = [1, 2, 3]               # possible: 1, 2, 3

    smiles_list = load_smiles(smiles_path, limit, seed)

    names = list(REPRESENTATIONS) if representation == "both" else [representation]
    all_ok = True
    for name in names:
        representation_cls, vectorizer_cls = REPRESENTATIONS[name]

        for test in sorted(tests):             # always 1, 2, 3 (see note on START/END above)
            if test == 1:
                ok = test_direct(representation_cls, smiles_list)
                print("\n")
            elif test == 2:
                ok = test_vectorized(vectorizer_cls, smiles_list, False, batch_size)
                print("\n")
            else:
                ok = test_vectorized(vectorizer_cls, smiles_list, True, batch_size)
                print("\n")
            all_ok = all_ok and ok

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()