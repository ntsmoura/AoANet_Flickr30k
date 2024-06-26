"""
This code is responsible for translate flickr_30k dataset json to portuguese
It runs over the JSON translating raw sentences and splitting translated tokens
It keeps track of already done translations and avoid to repeat them

To run this translator you need to set a groq api key on environment variables.
"""

import asyncio
import json
import os
import re
from copy import copy
from json import JSONDecodeError
from pathlib import Path

from groq import Groq

from translation.base_translator import BaseTranslator
from tqdm import tqdm

from translation.config import settings


class InvalidSentencesQuantity(Exception):
    def __init__(self, msg=None):
        if not msg:
            msg = "Quantity of sentences is different from expected..."
        super().__init__(msg)


class InvalidSentenceSize(Exception):
    def __init__(self, msg=None):
        if not msg:
            msg = "Size of any sentence is invalid..."
        super().__init__(msg)


class InvalidAnswer(Exception):
    def __init__(self, msg=None):
        if not msg:
            msg = "Invalid Answer..."
        super().__init__(msg)


class GroqTranslate(BaseTranslator):
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
        Groq Translator

        :param source_json: The path to source flickr dataset json.
        :param checkpoint_path: Where the translator sould save the checkpoint json.
        :param output_path: Where the translated json should be saved.
        :param source_language: The source language of translation.
        :param dest_language: The destination language of translation.
        """
        super().__init__(
            translator_identifier="groq",
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            source_json=source_json,
            source_language=source_language,
            dest_language=dest_language,
        )

        self.base_prompt = (
            f"TRANSLATE THE FOLLOWING SENTENCES FROM {self.source_language.upper()} TO {self.dest_language.upper()}:"
            f"\n\nREPLACE_THIS_WITH_SENTENCES"
            "\n\nAND RETURN THEM INTO THE SAME JSON STRUCTURE REPLACING THE ORIGINAL SENTENCE FOR "
            "THE TRANSLATED ONE.\nSTRICT RULES:\n"
            "DO NOT ANSWER ANYTHING BESIDES THE JSON\nRETURN A VALID JSON THAT CAN BE"
            " PARSED BY PYTHON json.loads FUNCTION\nDO NOT MODIFY THE JSON STRUCTURE.\n"
            'DO NOT USE \\" TO REPRESENT QUOTES OF JSON DELIMITERS.'
        )

        self.max_sentence_batches = 30
        self.groq_client = Groq(api_key=settings.api_keys.GROQ_API_KEY)
        self.requests_made = 0
        # llama3-8b-8192 was chosen because the balance beetween tokens per minute/answer quality
        self.base_llm_model = "llama3-8b-8192"

    async def translate_sentences(self):
        translating_img_ids = []
        translating_sentences = []
        images = self._flickr_source_json["images"]
        infos_dict = {}

        for image in tqdm(images):
            if self.requests_made >= 14390:
                # Groq (llama3-8b-8192) has a limit of 14400 requests per day
                raise Exception("Requests per day limit reached...")

            image_id = image["imgid"]
            infos_dict[image_id] = image
            if str(image_id) not in self._checkpoint_dictionary:
                translating_img_ids.append(image_id)
                prompt = copy(self.base_prompt)
                joined_sentences = json.dumps(
                    {
                        index: sentence["raw"]
                        for index, sentence in enumerate(image["sentences"])
                    }
                )
                prompt = prompt.replace("REPLACE_THIS_WITH_SENTENCES", joined_sentences)
                translating_sentences.append(prompt)

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

                # Groq (llama3-8b-8192) has a limit of 30 requests per minute
                await asyncio.sleep(65)

    @staticmethod
    def write_wrong_answer_to_disk(llm_original_anwser):
        with open("translation_data/llm_invalid_answers.txt", "a") as error_file:
            error_file.write(llm_original_anwser + "\n")

    def assert_valid_answer(self, answer):
        found_error = False

        original_answer = copy(answer)

        if answer[-1] != "}":
            if "}" not in answer:
                answer = answer[:-1] + "}"
            else:
                char_position = answer.find("}")
                if char_position != -1:
                    answer = answer[: char_position + 1]
            found_error = True

        if answer[0] != "{":
            if "{" not in answer:
                answer = "{" + answer[1:]
            else:
                char_position = answer.find("{")
                if char_position != -1:
                    answer = answer[char_position:]
            found_error = True

        if len(answer) < 100:
            raise InvalidAnswer()

        answer = answer.strip()
        answer = answer.replace("\n", "")

        if "\\'" in answer:
            found_error = True
            answer = answer.replace("\\'", "'")

        if match := re.match(r"{,|,{|,}|},", answer):
            found_error = True
            match_text = match.group(0)
            answer = answer.replace(match_text, match_text.strip(","))

        if found_error:
            self.write_wrong_answer_to_disk(original_answer)

        return answer

    async def send_sentences_to_api(self, sentences_matrix: [str]) -> (int, [str]):
        """
        Send sentences to groq llm api and returns translated sentences
        """

        def parse_response(response_text_):
            nonlocal translated_sentences_matrix
            json_result_ = json.loads(response_text_)
            if len(json_result_.keys()) != 5:
                raise InvalidSentencesQuantity()
            sentences_ = list(json_result_.values())
            if any(len(sentence_) <= 10 for sentence_ in sentences_):
                raise InvalidSentenceSize()
            translated_sentences_matrix.append(list(json_result_.values()))

        translated_sentences_matrix = []
        for prompt in sentences_matrix:
            while True:
                # Tries until get a valid json or another error occurs
                try:
                    reponse = self.groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        model=self.base_llm_model,
                    )

                    self.requests_made += 1

                    if self.requests_made >= 14400:
                        raise Exception("Requests per day limit reached...")

                    original_response_text = reponse.choices[0].message.content
                    response_text = self.assert_valid_answer(
                        copy(original_response_text)
                    )
                    try:
                        parse_response(response_text)
                    except JSONDecodeError:
                        self.write_wrong_answer_to_disk(original_response_text)
                        # Try to parse again replacing \" for ", that's a common llm error
                        response_text = response_text.replace('\\"', '"')
                        parse_response(response_text)
                    break
                except (
                    JSONDecodeError,
                    InvalidSentencesQuantity,
                    InvalidAnswer,
                    InvalidSentenceSize,
                ):
                    continue

        return translated_sentences_matrix


async def main():
    checkpoint_path = Path(__file__).parent / "translation_checkpoint"
    data_path = Path(__file__).parent / "translation_data"

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    groq_translator = GroqTranslate(
        checkpoint_path=checkpoint_path, output_path=data_path
    )

    await groq_translator.translate_sentences()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
