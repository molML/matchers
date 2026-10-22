import re
from typing import List
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

RDLogger.DisableLog("rdApp.*")

_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation": rf"(\[[^\]]+]|{_ELEMENTS_STR}|"     # All characters included within a braket [ XX ]
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",    # All possible characters that can be used inclusive brackets
    "leading_mass": rf"\[\d+({_ELEMENTS_STR})",     # Ions or brackets elements [ X ...
    "solo_element": rf"\[({_ELEMENTS_STR})\]",      # Elements in [ X ]
    "rings": r"\%\d{2}|\d",     # Percent with two digits afterwards
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def clean_smiles(
    smiles: str,
    remove_salt=True,
    desalt=False,
    uncharge=True,
    sanitize=True,
    remove_stereochemistry=True,
    explicit_h=True,
    to_canonical=True,
):
    """
    Cleaning of a simple SMILES string. Used in the function below.
    """
    if remove_salt and is_salt(smiles):
        return None

    if remove_stereochemistry:
        smiles = eliminate_stereochemistry(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(mol)
    if mol is None:
        return None
    salt_remover = SaltRemover()
    uncharger = rdMolStandardize.Uncharger()
    if desalt:
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    if explicit_h:
         mol = Chem.RemoveHs(mol)
    if uncharge:
        mol = uncharger.uncharge(mol)
    if sanitize:
        sanitization_flag = Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
        )
        # SANITIZE_NONE is the "no error" flag of rdkit
        if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None

    return Chem.MolToSmiles(mol, canonical=to_canonical)

def is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def is_valid_batch(smiles_batch: List[str]) -> List[bool]:
    return [is_valid(smiles) for smiles in smiles_batch]

def eliminate_stereochemistry(smiles, replace_dict=None):
    """ 
    Elimination of Stereochemistry.
    Arguments:
    smiles_list: list of SMILES 
    replace_dict: dict for replace of strings
    Return: 
    replaced SMILES-list 
    """

    if not replace_dict:
        replace_dict = {'[C@H]':'C', '[C@@H]':'C', '[C@]':'C', '[C@@]':'C', 
                        '/C' : 'C', 'C/' : 'C', 
                        '[P@]': 'P', '[P@@]': 'P', '[P@@+]': '[P+]', '[P@+]': '[P+]', '[P@H]': 'P', '[P@@H]': 'P',
                        '[N@@]': 'N', '[N@]': 'N', '[N@+]': '[N+]', '[N@@+]': '[N+]', 
                        '[S@@]': 'S', '[S@]': 'S', '[S@@+]': '[S+]', '[S@+]': '[S+]'}
    
    for key in replace_dict.keys():
        if key in smiles:
            smiles = smiles.replace(key, replace_dict[key])

    return smiles

def is_salt(smiles: str, negate_result=False) -> bool:
    is_salt = "." in set(smiles)
    if negate_result:
        return not is_salt
    return is_salt

def desalt(smiles: str, to_canonical=False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid input chemical")
    remover = SaltRemover()
    desalted = remover.StripMol(mol, dontRemoveEverything=True)
    return Chem.MolToSmiles(desalted, canonical=to_canonical)

def contains_solo_element(smiles):
    return len(_RE_PATTERNS["solo_element"].findall(smiles)) > 0

def contains_atomic_mass(smiles):
    return len(_RE_PATTERNS["leading_mass"].findall(smiles)) > 0

def find_n_rings(smiles: str) -> int:
    '''
    Finds all higher-numbered rings structures and returns the amount of rings in the molecule.
    arguments: SMILES string
    return: Number 
    '''
    return len(_RE_PATTERNS["rings"].findall(smiles)) // 2      # Rings start and end with the same number, only count the number once.

def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    ''' 
    Divides the SMILES string and gives each element and special characters.
    arguments: SMILES string and segment_sq
    return: 
    list of tokenized SMILES string
    '''
    regex = _RE_PATTERNS["segmentation_sq"]
    if not segment_sq_brackets:
        regex = _RE_PATTERNS["segmentation"]
    return regex.findall(smiles)