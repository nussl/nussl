import pytest
from nussl import separation

def test_spatial_clustering(scaper_folder):
    dataset = datasets.Scaper(scaper_folder)
    item = dataset[5]

    
