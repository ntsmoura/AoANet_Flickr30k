"""
This code is responsible for translate flickr_30k dataset json to portuguese
It runs over the JSON translating raw sentences and splitting translated tokens
It keeps track of already done translations and avoid to repeat them

To run this translator you need a google cloud account and to set the cloud credentials on google cloud cli.
"""

import asyncio
import os
from pathlib import Path

from google.cloud import translate

from translation.base_translator import BaseTranslator
from tqdm import tqdm


class GoogleCloudTranslate(BaseTranslator):
    def __init__(
        self,
        checkpoint_path: Path,
        output_path: Path,
        source_json: Path = Path(__file__).parent.parent
        / "data"
        / "flickr30k_dataset.json",
        source_language: str = "en-US",
        dest_language: str = "pt-BR",
    ):
        """
        GoogleCloud Translator

        :param source_json: The path to source flickr dataset json.
        :param checkpoint_path: Where the translator sould save the checkpoint json.
        :param output_path: Where the translated json should be saved.
        :param source_language: The source language of translation.
        :param dest_language: The destination language of translation.
        """
        super().__init__(
            translator_identifier="googlecloud",
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            source_json=source_json,
            source_language=source_language,
            dest_language=dest_language,
        )
        self.max_sentence_batches = 50

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
                translating_sentences.extend(
                    [[sentence["raw"] for sentence in image["sentences"]]]
                )

            if (
                len(translating_img_ids) >= self.max_sentence_batches
                or image_id == images[-1]["imgid"]
            ):
                results = await self.send_sentences_to_api(translating_sentences)
                parse_coros = []
                for index, result in enumerate(results):
                    image_id = translating_img_ids[index]
                    translated_sentences = result
                    translation_dict = infos_dict[image_id]
                    for sentid, sentence in enumerate(translation_dict["sentences"]):
                        sentence["raw"] = translated_sentences[sentid]
                        sentence["tokens"] = (
                            translated_sentences[sentid].strip(". ").lower().split()
                        )
                    checkpoint_data = {image_id: "ok"}
                    parse_coros.append(
                        self.append_translated_sentences_to_output(
                            translation_dict, checkpoint_data
                        )
                    )

                await asyncio.gather(*parse_coros)
                translating_img_ids = []
                translating_sentences = []

    async def send_sentences_to_api(self, sentences_matrix: [[str]]) -> (int, [str]):
        """
        Send sentences to googlecloudtranslate api and returns translated sentences
        """
        project_id = "xenon-effect-420413"  # You need to set your own project id

        client = translate.TranslationServiceClient()

        location = "global"

        parent = f"projects/{project_id}/locations/{location}"

        translated_sentences_matrix = []
        for sentence_list in sentences_matrix:
            response = client.translate_text(
                request={
                    "parent": parent,
                    "contents": sentence_list,
                    "mime_type": "text/plain",  # mime types: text/plain, text/html
                    "source_language_code": "en-US",
                    "target_language_code": "pt-BR",
                }
            )

            translated_sentences_matrix.append(
                [translation.translated_text for translation in response.translations]
            )

        return translated_sentences_matrix


async def main():
    checkpoint_path = Path(__file__).parent / "translation_checkpoint"
    data_path = Path(__file__).parent / "translation_data"

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    googlecloud_translator = GoogleCloudTranslate(
        checkpoint_path=checkpoint_path, output_path=data_path
    )

    await googlecloud_translator.translate_sentences()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
