from .builders import *
from .parser import build_parser

import json
import pkg_resources

json_loader = lambda path: json.loads(
        pkg_resources.resource_string(__name__, path)
    )

subparsers_json = json_loader("subparsers.json")
defaults = {}

for subparser_name, metadata in subparsers_json.items():
    defaults[subparser_name] = json_loader(metadata['defaults_path'])