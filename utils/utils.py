import json
import pathlib

def get_package_path():
    return str(pathlib.Path(__file__).parent.parent.resolve())

package_path = get_package_path()

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)