###
# My normalisation code is going to operate on the entire file of training text,
# and go through every single word
###
import re
import unicodedata

punctuation_to_remove = [
    ".", ",", "!", "?", ";", ":",  # punctuation
    "'", '"', "`", "’", "“", "”",  # quotes/apostrophes
    "(", ")", "[", "]", "{", "}",  # brackets
    "-", "—", "_",                 # dashes/underscores
    "/", "\\",                     # slashes
    "@", "#", "$", "%", "^", "&", "*",  # symbols
    "+", "=", "<", ">", "|", "~"
]

def normalise_cap_space(text: str) -> str:
    """
    Function that returns the text lower-cased -
    and stripped of all leading/tailing whitespace
    """
    return text.lower().strip()

def normalise_punc(sentences):
    """
    Function that passes over all chars in text -
    to remove unwanted punctuation
    """
    # Apply a forward pass looking for punctuation
    i = 0
    for sentence in sentences:

        result = []
        for char in sentence:

            if char in punctuation_to_remove:
                result.append(" ")
            else:
                result.append(char)
        
        # strip any excess whitespace
        sentences[i] = "".join(result).strip()
        i += 1
    return sentences


def split_sentences(text):
    """
    A function that splits the text into sentences per line -
    by detecting either of the following punctuation:
    . ! ? 
    and with a following whitespace
    """
    # first replace newline chars w space to account for titles
    text = text.replace('\n', ' ')

    sentences = re.split(r'[.!?]+\s*', text)

    return sentences

def normalise_diacritics(text):
    """
    A function that replaces diacritic symbols with their standard -
    ASCII character, eg: é -> e
    """
    normalized = unicodedata.normalize('NFKD', text)
    return(''.join(
        ch for ch in normalized
        if not unicodedata.combining(ch)
    ))

def normalise_numbers(text):
    """
    Function that normalises the numbers in the text.
    Method looks for numbers, and replaces with a 0-token to-
    represent numbers
    """
    result = []
    i = 0
    while i < len(text):
        if text[i].isdigit():
            # keep scanning for nums
            while i < len(text) and text[i].isdigit():
                i +=1
            result.append("0")
        else:
            result.append(text[i])
            i += 1
    return "".join(result)

def add_start_end_tokens(sentences):
    
    for i in range(len(sentences)):
        sentences[i] = "^^" + sentences[i] + "$"

    return sentences


def normalise_spaces(sentences):
    """
    A function that just strips the sentence of any spaces we
    may have added during processing
    """
    for i in range(len(sentences)):
        sentences[i] = re.sub(r'\s+', ' ', sentences[i]).strip()
    #final filter out any empty sentences
    sentences = [s for s in sentences if s]

    return sentences

def normalise_text(text):
    """
    Function that Normalises the entire text
    """
    # first lower case the text
    text = text.lower()

    #normalise for numbers
    text = normalise_numbers(text)

    #normalise diacritics
    text = normalise_diacritics(text)

    # split the sentences
    sentences = split_sentences(text)

    #remove punctuation
    sentences = normalise_punc(sentences)

    #strip any additave spaces
    sentences = normalise_spaces(sentences)

    # add start stop tokens
    sentences = add_start_end_tokens(sentences)

    return sentences

if __name__ == "__main__":
    #load example text
    with open("../data/raw/train.en.txt") as f:
        text = f.read()

    sentences = normalise_text(text)
    print(sentences)
