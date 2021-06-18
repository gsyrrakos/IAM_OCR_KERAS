from autocorrect import Speller
from textblob import TextBlob
from textblob import Word


def correct_sentence(line):
    new_line = TextBlob(line).correct()

    return new_line
