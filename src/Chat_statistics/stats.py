import json
from pathlib import Path
from typing import Union

import arabic_reshaper
from bidi.algorithm import get_display
from hazm import Normalizer, word_tokenize
from loguru import logger
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    """Generates chat statistics from a chat json file
    """
    def __init__(self, json_file: Union[str, Path]):
        """
        Args:
            json_file: path to telegram export json file
        """

        # load chat data
        logger.info(f"Loading chat data from {json_file}")
        with open(json_file) as f:
            self.chat_data = json.load(f)

        self.normalizer = Normalizer()

        # load stop words
        logger.info(f" Loading stop words from {DATA_DIR / 'stopwords.txt'}")
        stop_words = open(DATA_DIR / 'stopwords.txt').readline()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.normalizer.normalize, stop_words))

    def generate_word_cloud(self, output_path: Union[str, Path]):
        """Genarates word cloud from chat data

        Args:
            output_path: path to output directory for word cloud image
        """
        logger.info('Loading text content...')
        txt_cont = ''
        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = word_tokenize(msg['text'])
                tokens = list(filter(lambda item: item not in self.stop_words,
                              tokens))

                txt_cont += f" {' '.join(tokens)}"

        # Normalize, reshape for final word cloud
        txt_cont = self.normalizer.normalize(txt_cont)
        txt_cont = arabic_reshaper.reshape(txt_cont[:3_000])
        txt_cont = get_display(txt_cont)

        logger.info('Generating word cloud...')
        wordcloud = WordCloud(
          width=1200, height=600,
          font_path=str(Path(DATA_DIR) / 'NotoNaskhArabic-Regular.ttf'),
          background_color='white',
          max_font_size=250
          ).generate(txt_cont)

        logger.info("saving word cloud to {output_path}")
        wordcloud.to_file(str(Path(output_path) / 'Wordcloud.png'))


# testing code
if __name__ == '__main__':
    chat_stat = ChatStatistics(json_file=DATA_DIR / 'result.json')
    chat_stat.generate_word_cloud(output_path=DATA_DIR)
    print('Done!')
