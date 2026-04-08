import numpy as np

def language_predictor(text, language_models):
    """
    A Function that predicts the language of a text string using -
    the perplexity of the text under each language model 
    """
    perplexities = []
    for lm in language_models:
        perplexities.append(lm.perplexity(text))
    best_lm_idx = np.argmax(perplexities)
    return(language_models[best_lm_idx])


def evaluate_language_id(dataset, language_models):
    """
    Compute accuracy and possibly collect mistakes for analysis.
    """
    pass

def load_test_sentences(path):
    """
    Read labeled test data file into structured examples.
    """
    pass

if __name__ == "__main__":
    pass