from src.poem_generator import PoemModel

URL = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Poem_models/gpt2_en_poems/gpt2_en_poems.json"
URL2 = "https://minio.lab.sspcloud.fr/paultoudret/ensae-reproductibilite/Phoetry/Poem_models/gpt2_fr_haiku/gpt2_fr_haiku.json"

Poem_gen = PoemModel(URL)
