"""
This code is responsible for translate flickr_30k dataset json to portuguese
It runs over the JSON translating raw sentences and splitting translated tokens
It keeps track of already done translations and avoid to repeat them
"""

from pathlib import Path

from translation.base_translator import BaseTranslator


class LibreTranslate(BaseTranslator):
    def __init__(
        self,
        checkpoint_path: Path,
        output_path: Path,
        source_json: Path = Path(__file__).parent.parent / "data" / "flickr30k_dataset.json",
        source_language: str = "en",
        dest_language: str = "pt",
    ):
        """
        LibreTranslate Translator

        :param source_json: The path to source flickr dataset json.
        :param checkpoint_path: Where the translator sould save the checkpoint json.
        :param output_path: Where the translated json should be saved.
        :param source_language: The source language of translation.
        :param dest_language: The destination language of translation.
        """
        super().__init__(
            translator_identifier="libretranslate",
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            source_json=source_json,
            source_language=source_language,
            dest_language=dest_language,
        )
