
from normalisation import normalise_text

def calculate_counts(train_sentences):
    """
    Function that slides over the entire training corpus -
    computes the total number of times a letter x follows previous two
    """ 
    counts = {}
    for sentence in train_sentences:
        for i in range(len(sentence)-2):  # i is leftmost idx of sliding window
            prefix = sentence[i : i+2]
            char = sentence[i + 2]
            
            # if we've seen this prefix incr, otherwise initialise
            if prefix in counts:
                if char in counts[prefix]:
                    counts[prefix][char] += 1   # incr
                else:
                    counts[prefix][char] = 0    # init
            else:
                counts[prefix] = {}             # init
    return counts

def calculate_history_counts(counts):
    """
    Function that outputs a dictionary of the total counts
    """ 
    roll_sum = 0
    history_counts = {}
    for prefix in counts:
        history_counts[prefix] = 0
        for char in counts[prefix]:
            history_counts[prefix] = history_counts[prefix]+ counts[prefix][char]

    return history_counts


class TrigramModel:

    def __init__(self):
        self.counts = {}            # {"th":{ e : 500, o : 100, ...}}
        self.history_counts = {}    # {"th": 800, "tr": 200, ... }
        self.vocab = set()          # {"a", "b", ...}

    def fit(self, train_sentences, regularisation=None):
        """Estimate trigram counts/probabilities from training texts."""
        
        self.counts = calculate_counts(train_sentences)

        pass

    def get_probability(self, history, char):
        """Return p(char | history)."""
        pass

    def score_text(self, text):
        """Return log-probability or total score for a text sequence."""
        pass

    def perplexity(self, texts):
        """Compute perplexity on a list of texts."""
        pass

    def generate(self, max_length=200, seed_text=None):
        """Generate character text from the model."""
        pass

    def save(self, path):
        """Save model to disk."""
        pass



if __name__ == "__main__":
    with open("../data/raw/train.en.txt") as f:
        text = f.read()

    sentences = normalise_text(text)

    print(calculate_counts(sentences))

    