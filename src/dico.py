"""
Only a dico with labels associated to a key
"""
import json
import requests


class Label_dico():
    """
    A simple class derived from dict() to deal with labels
    """
    dico: dict
    """
    A dico with :
    key --> theme
    list --> labels
    """
    URL: str
    """
    The URL of labels stocked on S3 SSP CLOUD
    Not sensitive --> free access is ok
    """

    def __init__(self):
        """
        Initialize the dico by charging the file labels.json
        """
        self.dico = dict()
        self.URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Datasets/labels.json"

        # Get the response of the requested URL
        response = requests.get(self.URL)

        # Vérification du statut de la réponse
        if response.status_code == 200:  # Positive request
            self.dico = response.json()
        else:
            print(f"Erreur lors du téléchargement du fichier : {response.status_code}")

    def add_label(self, key: str, labels: [str]):
        """
        Used to add a theme (key) and its associated labels (list of strings)
        Return nothing
        """
        if key in self.dico.keys():
            print(f"this key -{key}- is already used\n"
            "you must give another key")
        else:    
            self.dico[key] = labels
            print(f"-{key}- and its labels have been added to dico\n"
            "In order to record the dico you should used the '.export_dico()' method")

    def export(self):
        """
        NOT READY YET
        NEED TO STOCK THE JSON ON S3 SSP CLOUD
        """
        with open('./data/dico.json', 'w') as json_file:
            json.dump(self.dico, json_file)
            
        print("The dico has been well exported")

