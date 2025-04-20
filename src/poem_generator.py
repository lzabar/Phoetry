import requests
import os
import time
from src.my_log import get_logger
from transformers import GPT2Tokenizer, GPT2LMHeadModel


logger = get_logger(name=__name__)


class PoemModel:
    """
    A class to initialize a model and use it
    """
    URL: str
    """
    The URL to get the info of the model
    """
    dico: dict
    """
    A dictionnary to stock informations about the model
    """
    local_dir: str
    """
    The address of the local directory where to stok the model
    """
    name: str
    """
    The name of the fine tuned model. Unique to every model
    """
    model: GPT2LMHeadModel
    """
    Type of model used
    """
    tokenizer: GPT2Tokenizer
    """
    Tokenizer used for GPT2 model
    """

    def __init__(self, URL):
        """
        Initialize the model by checking the model is available thanks to its URL.
        Input :
            - URL to download the model
        Initialize also:
            - Name of the model
            - Local directory where to download the model
            - Model and Tokenizer
        """
        start = time.time()

        self.URL = URL
        logger.info(f"Initialization : Start --> {URL}")

        response = requests.get(self.URL)

        # Checking status
        if response.status_code == 200:     # Positive request
            self.dico = response.json()
            
        else:
            logger.error(f"Download : Fail --> dico : {response.status_code}")
            return None

        # Checking the all keys are in the config file
        if self.check_dico() is False:      # Not all keys in self.dico
            logger.error("Config : Fail --> keys")
            return None
        else:
            logger.info("Config : Success --> keys")
            self.name = self.dico['name']
            self.local_dir = './trained_model/' + self.name + '/'    # where the model will be stored localy
            os.makedirs(self.local_dir, exist_ok=True)

            logger.info(f"local dir set to {self.local_dir}")

            bucket_URL = self.dico['URL']

            # We download every file in the list
            i = 1                                              # only for printing
            for file in self.dico['files']:
                if os.path.exists(self.local_dir + file):      # checking that file is  already downloaded
                    logger.info(f"Already downloaded : [{i}/{len(self.dico['files'])}] Success --> {file}")
                    i += 1
                else:
                    resp = requests.get(bucket_URL + self.name + '/' + file)

                    if resp.status_code == 200:
                        st = time.time()
                        with open(self.local_dir + file, "wb") as f:
                            f.write(resp.content)
                            es = time.time()
                            logger.info(f"Download : [{i}/{len(self.dico['files'])}] Success in {round(es - st, 2)}--> {file}")
                            i += 1
                    else:
                        logger.error(f"Download : Fail --> {file} // error : {resp.status_code}")
                        return None

        # Try to initialize the model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.local_dir)
            self.model = GPT2LMHeadModel.from_pretrained(self.local_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.model.eval()
            end = time.time()
            delta = round(abs(end - start), 1)

            logger.info(f'Initialization : Complete in {delta} sec --> Model is running')

        except Exception as e:
            raise RuntimeError(f"Error loading model from {self.local_dir}: {e}")

    def check_dico(self):
        """
        Basic function to ensure that the config file representing the model
        has every key it needs.
        This function has to be changed (var keys) if the structure of config file
        changes.
        """
        keys = [
            'description',
            'name',
            'version',
            'status',
            'date_of_release',
            'URL',
            'files'
        ]

        if keys == list(self.dico.keys()):
            return True
        else:
            logger.debug(f'keys should be {keys}', end='\n')
            logger.debug(f'but found {self.dico.keys()}')
            return False

    def generate_poem(self, theme: str, poem_type: str):
        """
        Generate a poem based on a theme and poem type.
        The parameters from each type are automatically loaded from the JSON config.
        """
        params = self.dico.get("arg_poem_gpt", {})
        if poem_type not in params:
            raise ValueError(f"Poem type '{poem_type}' not found in configuration file.")

        args = params[poem_type]

        inputs = self.tokenizer.encode(
            theme,
            return_tensors="pt",
            )
        logger.info("Encoding : Success --> inputs")

        # Generate text
        output = self.model.generate(
            inputs,
            do_sample=True,
            **args
        )
        logger.info("Generating : Success --> outputs")

        # Decode and print the poem
        poem = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # do not include theme at the start of the poem
        poem = poem[len(theme):].strip()

        logger.info("Decoding : Succes --> poem")
        logger.info(f"Length of poem : {len(poem)}")
        return poem
