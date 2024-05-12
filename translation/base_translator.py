import json
import pickle
from asyncio import Lock
from pathlib import Path


class BaseTranslator:
    def __init__(self, translator_identifier: str, checkpoint_path: Path, output_path: Path, source_json: Path = Path(__file__).parent.parent / "data" / "flickr30k_dataset.json", source_language: str = 'en', dest_language: str = 'pt'):
        """
        Translator base class

        :param translator_identifier: Identifies the translator, applied to output filenames (checkpoint and json).
        :param source_json: The path to source flickr dataset json.
        :param checkpoint_path: Where the translator sould save the checkpoint json.
        :param output_path: Where the translated json should be saved.
        :param source_language: The source language of translation.
        :param dest_language: The destination language of translation.
        """
        self.translator_identifier = translator_identifier
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.source_json = source_json
        self.source_language = source_language
        self.dest_language = dest_language
        self._flickr_source_json = None
        self._flickr_dest_json = None
        self._checkpoint_dictionary = dict()
        self.checkpoint_lock = Lock()
        self.output_lock = Lock()

    def read_source_json(self):
        """
        Reads the source dataset json.
        """
        with open(self.source_json) as file:
            self._flickr_source_json = json.load(file)

    def save_checkpoint(self):
        """
        Saves dataset translation checkpoint.
        """
        async with self.checkpoint_lock:
            with open(self.checkpoint_path / f"{self.translator_identifier}_flicker30k_checkpoint.json", "w+") as file:
                source_dict = json.load(file)
                source_dict.upload(self._checkpoint_dictionary)
                json.dump(source_dict, file)
    def load_checkpoint(self):
        """
        Loads dataset translation checkpoint.
        """
