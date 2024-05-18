import json
import os.path
from asyncio import Lock
from copy import copy
from pathlib import Path


class BaseTranslator:
    def __init__(
        self,
        translator_identifier: str,
        checkpoint_path: Path,
        output_path: Path,
        source_json: Path = Path(__file__).parent.parent / "data" / "flickr30k_dataset.json",
        source_language: str = "en",
        dest_language: str = "pt",
    ):
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
        self.output_lock = Lock()
        self.max_sentence_batches = 25

        self.load_checkpoint()
        self.read_source_json()
        self.create_or_load_ouput_json()

    def read_source_json(self):
        """
        Reads the source dataset json.
        """
        with open(self.source_json) as file:
            self._flickr_source_json = json.loads(file.read())

    def save_checkpoint(self, checkpoint_data: dict):
        """
        Saves dataset translation checkpoint.

        :param checkpoint_data: Data to be saved in checkpoint dictionary.
        """
        with open(self.checkpoint_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k_checkpoint.json", "w+") as file:
            self._checkpoint_dictionary.update(checkpoint_data)
            dumped_json = json.dumps(self._checkpoint_dictionary)
            file.write(dumped_json)

    def load_checkpoint(self):
        """
        Loads dataset translation checkpoint.
        """
        try:
            with open(
                self.checkpoint_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k_checkpoint.json",
                "r",
            ) as file:
                self._checkpoint_dictionary = json.loads(file.read())
        except IOError:
            pass

    async def translate_sentences(self):
        """
        Split dataset images sentences into batches and translate them using the specified method.
        """
        raise NotImplementedError

    async def append_translated_sentences_to_output(self, translation_dict: dict, checkpoint_data: dict):
        """
        Appends translated sentences information to output json.
        Saves checkpoint data.
        """
        old_flickr_dest_json = copy(self._flickr_dest_json)
        old_checkpoint = copy(self._checkpoint_dictionary)
        try:
            async with self.output_lock:

                with open(
                    self.output_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k.json", "r+"
                ) as file:
                    self._flickr_dest_json["images"].append(translation_dict)
                    dumped_json = json.dumps(self._flickr_dest_json)
                    file.write(dumped_json)

                self.save_checkpoint(checkpoint_data)
        except Exception:
            with open(
                self.output_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k.json", "r+"
            ) as file:
                dumped_json = json.dumps(old_flickr_dest_json)
                file.write(dumped_json)
            with open(self.checkpoint_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k_checkpoint.json", "w+") as file:
                dumped_json = json.dumps(old_checkpoint)
                file.write(dumped_json)

    def create_or_load_ouput_json(self):
        """Creates base output json or loads an existing one"""
        base_json = {"images": [], "dataset": "flickr30k"}
        if not os.path.exists(self.output_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k.json"):
            with open(
                self.output_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k.json", "w+"
            ) as file:
                json_string = json.dumps(base_json)
                file.write(json_string)

            self._flickr_dest_json = base_json
        else:
            with open(
                self.output_path / f"{self.translator_identifier}_{self.dest_language}_flicker30k.json", "r"
            ) as file:
                self._flickr_dest_json = json.loads(file.read())
