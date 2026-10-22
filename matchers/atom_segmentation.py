from sklearn.preprocessing import MinMaxScaler
from matchers.sequence.smiles_processing import segment_smiles
from matchers.sequence.smiles_processing import (
    contains_atomic_mass,
    contains_solo_element,
    find_n_rings,
)
from matchers.chemistry.chemistry import (
    get_periodic_table_properties,
    is_element,
)
import pandas as pd

__N_MAX_RINGS = 50

__SUPPORTED_ELEMENTS = {
    "C",
    "H",
    "O",
    "N",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "c",
    "n",
    "o",
    "s",
}

__PT_FEATURE_NAMES = [
    "quantum_number",
    "electron_number",
    "mass",
    "vdw_volume",
    "electronegativity",
    "polarizability",
    "ionization_potential",
    "covalent_radius",
]

__PT_FEATURES = {
    element: {
        prop_name: prop_value
        for prop_name, prop_value in properties.items()
        if prop_name in __PT_FEATURE_NAMES
    }
    for element, properties in get_periodic_table_properties(fillna=-1).items()
    if element in __SUPPORTED_ELEMENTS
}

__pt_scaler = MinMaxScaler()


def __standardize_pt_features():
    df_pt_features = pd.DataFrame(__PT_FEATURES).T
    df_normed = pd.DataFrame(
        __pt_scaler.fit_transform(df_pt_features),
        index=df_pt_features.index,
        columns=df_pt_features.columns,
    )
    return df_normed.to_dict(orient="index")


__ST_PT_FEATURES = __standardize_pt_features()
__ST_PT3_FEATURES = {
    element: {
        name: features[name] for name in ["mass", "vdw_volume", "electronegativity"]
    }
    for element, features in __ST_PT_FEATURES.items()
}


def __getattr__(attr_name):
    attributes = {
        "st_pt_features": __ST_PT_FEATURES,
        "st_pt3_features": __ST_PT3_FEATURES,
    }
    if attr_name not in attributes:
        raise AttributeError(f"module '{__name__}' has no attribute '{attr_name}'")

    return attributes[attr_name]


def is_supported_chemical(smiles, verbosity=False):
    def contains_unsupported_element(smiles):
        tokens = set(segment_smiles(smiles, segment_sq_brackets=True))
        elements = {token for token in tokens if is_element(token)}
        return len(elements.difference(__SUPPORTED_ELEMENTS)) > 0

    atomic_mass = contains_atomic_mass(smiles)
    solo_element = contains_solo_element(smiles)
    n_rings = find_n_rings(smiles)
    unsupported_element = contains_unsupported_element(smiles)
    if verbosity:
        if atomic_mass:
            print("MATCHERS doesn't support SMILES with atomic mass")
        if solo_element:
            print("MATCHERS doesn't support SMILES with non-charged metals")
        if n_rings > __N_MAX_RINGS:
            print(
                f"MATCHERS doesn't support chemical with more than {__N_MAX_RINGS} rings"
            )
        if unsupported_element:
            print(
                f"MATCHERS doesn't support chemicals that contain elements not in {__SUPPORTED_ELEMENTS}"
            )

    return (
        not atomic_mass
        and not solo_element
        and not n_rings > __N_MAX_RINGS
        and not unsupported_element
    )
