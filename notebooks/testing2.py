"""
Testing class LabelDico
"""

from src.label_dico import LabelDico


labels = LabelDico()

assert labels.URL == "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Datasets/labels.json"
assert labels.dico != {}

print(labels.dico.keys())
