"""
This code is responsible for translate flickr_30k dataset json to portuguese
It runs over the JSON translating raw sentences and splitting translated tokens
It keeps track of already done translations and avoid to repeat them
"""

import asyncio
import os
from pathlib import Path

from translation.base_translator import BaseTranslator

import httpx
from tqdm import tqdm


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

    async def translate_sentences(self):
        translating_img_ids = []
        translating_sentences = []
        images = self._flickr_source_json["images"]
        infos_dict = {}

        for image in tqdm(images):
            image_id = image["imgid"]
            infos_dict[image_id] = image
            if str(image_id) not in self._checkpoint_dictionary:
                translating_img_ids.append(image_id)
                translating_sentences.append("\n".join([sentence["raw"] for sentence in image["sentences"]]))

            if len(translating_img_ids) >= self.max_sentence_batches or image_id == images[-1]["imgid"]:
                results = await self.send_sentences_to_api(translating_sentences)
                parse_coros = []
                for index, result in enumerate(results):
                    image_id = translating_img_ids[index]
                    translated_sentences = result.split("\n")
                    translation_dict = infos_dict[image_id]
                    for sentid, sentence in enumerate(translation_dict["sentences"]):
                        sentence["raw"] = translated_sentences[sentid]
                        sentence["tokens"] = translated_sentences[sentid].strip(". ").lower().split()
                    checkpoint_data = {image_id: "ok"}
                    parse_coros.append(self.append_translated_sentences_to_output(translation_dict, checkpoint_data))

                await asyncio.gather(*parse_coros)
                translating_img_ids = []
                translating_sentences = []

    async def send_sentences_to_api(self, sentences: [str]) -> (int, [str]):
        """
        Send sentences to libretranslate api and returns translated sentences
        """
        json_data = {
            "q": sentences,
            "source": self.source_language,
            "target": self.dest_language,
            "format": "text",
            "api_key": "",
        }
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post("http://127.0.0.1:5000/translate", json=json_data)
            translated_sentences = response.json()["translatedText"]

        return translated_sentences

async def main():
    checkpoint_path = Path(__file__).parent / "translation_checkpoint"
    data_path = Path(__file__).parent / "translation_data"

    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    libre_translator = LibreTranslate(checkpoint_path=checkpoint_path, output_path=data_path)

    await libre_translator.translate_sentences()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())