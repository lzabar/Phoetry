"""
Testing class LabelDico
"""

from src.label_dico import LabelDico


def t_label_dico():
    labels = LabelDico()

    assert labels.URL == "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Datasets/labels.json"
    assert labels.dico != {}

    return True
