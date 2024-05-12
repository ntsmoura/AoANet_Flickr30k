"""
This code is responsible for translate flickr_30k dataset json to portuguese
It runs over the JSON translating raw sentences and splitting translated tokens
It keeps track of already done translations and avoid to repeat them
"""
from translation.base_translator import BaseTranslator


class LibreTranslate(BaseTranslator):
    pass