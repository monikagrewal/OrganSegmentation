import os
from typing import Dict, List
from utils.cache import RuntimeCache

def create_subfolders(root_folder: str, 
                    foldernames: Dict,
                    cache: RuntimeCache = None) -> None:
    
    for name, value in foldernames.items():
        folderpath = os.path.join(root_folder, value)
        os.makedirs(folderpath, exist_ok=True)
        if cache is not None:
            cache.__setattr__(name, folderpath)
