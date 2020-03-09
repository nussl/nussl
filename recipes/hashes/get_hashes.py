from nussl import efz_utils
import json

with open('musdb_hashes.json', 'w') as f:
    hashes = {}
    hash_ = efz_utils._hash_directory('/home/data/musdb/raw/musdb_unzip')
    hashes['musdb'] = hash_
    json.dump(hashes, f, indent=4)

with open('wham_hashes.json', 'w') as f:
    hashes = {}
    wav8k_hash = efz_utils._hash_directory('/home/data/wham/wav8k')
    wav16k_hash = efz_utils._hash_directory('/home/data/wham/wav16k')

    hashes['WHAM'] = {
        'wav8k': wav8k_hash,
        'wav16k': wav16k_hash
    }
    json.dump(hashes, f, indent=4)
