import re
from itertools import chain


class syllable_break():
    def __init__(self):

        self.MY_SYLLABLE_PATTERN = re.compile(
            r'(?:(?<!္)([\U00010000-\U0010ffffက-ဪဿ၊-၏]|[၀-၉]+|[^က-၏\U00010000-\U0010ffff]+)(?![ှျ]?[့္်]))',
            re.UNICODE)

        self.ENG_MY_SPLIT_PATTERN = re.compile(
            r'[https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9_][a-zA-Z0-9-_]+[a-zA-Z0-9_]\.[^\s]{2,}|www\.[a-zA-Z0-9_][a-zA-Z0-9-_]+[a-zA-Z0-9_]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9_]+\.[^\s]{2,}]+|[\u2707-\u27B0]+|[\U00010000-\U0010ffff]+|[က-ဪဿ၊-၏^က-၏ ]+|[\/0-9a-zA-Z_\.@\-]+|[""!\"#$%&\'()*\+,-./:;<=>?@\[\\\]^_`{|}~]+',
            re.UNICODE)
        self.REPLACE_DICT = {
                                " ' ": "'",
                                " '": "'",
                                " ,": ",",
                            }
                            
    def replace_all(self, text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def syllable_break(self, text):
        return self.MY_SYLLABLE_PATTERN.sub(r'𝕊\1', text).strip('𝕊').split('𝕊')

    def separate_eng_mm(self, text):
        return self.ENG_MY_SPLIT_PATTERN.findall(text)

    def syllable_break_both(self, text):
        '''syllable_break_both, syllable_break_list for correct syllable segmentation'''
        return list(
            chain.from_iterable([
                self.syllable_break(i) for i in self.separate_eng_mm(text)
                if i != ' ' or i != ''
            ]))

    def syllable_break_list(self, text):
        '''Parameters:
        input_value: list
        Returns: syllable values with no spaces
        e.g. [['09950367221', 'တစ်', 'SwanAung', 'car', 'တယ်', 'OK', '$', '😁', '😂'],
                    ['slkfjlskfj', 'car', 'စာ', 'အုပ်', 'sfsfd']] '''
        words = [self.syllable_break_both(data) for data in text]
        filtered_words = [
            list(filter(lambda word: word.strip(), msg)) for msg in words
        ]
        syl_sent = [' '.join(data) for data in filtered_words]
        syl_sent = [self.replace_all(sent, self.REPLACE_DICT) for sent in syl_sent]
        return syl_sent