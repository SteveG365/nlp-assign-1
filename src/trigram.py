
from normalisation import normalise_text
import numpy as np
import random

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


    def fit(self, train_sentences):
        """Estimate trigram counts/probabilities from training texts."""
        
        self.counts = calculate_counts(train_sentences)
        self.history_counts = calculate_history_counts(self.counts)
        

    def get_probability(self, prefix, char):
        """
        Fetches the relevant probability p(char | prefix) and -
        uses Kneser Ney smoothing to steal from the rich and give to the poor
        """
        count = self.counts[prefix].get(char, 0)
        total = self.history_counts.get(prefix, 0)

        if total == 0:
            return 1e-12

        prob = count / total
        return max(prob, 1e-12)

    def perplexity(self, sentences):
        """Compute perplexity on a list of texts."""
        log_prob_sum = 0
        n_predictions = 0
        for sentence in sentences:
            # sliding 3 window over whole text

            # compute p(char | prefix) using Kneser Ney Smoothing

            for i in range(len(sentence)-2):
                prefix = sentence[i:i+2]
                char = sentence[i + 2]

            prob = self.get_probability(prefix, char)

            # avoid log(0)
            if prob == 0:
                return float("inf")

            log_prob_sum += np.log(prob)
            n_predictions += 1

        if n_predictions == 0:
            return float("inf")

        return np.exp(-log_prob_sum / n_predictions)
        

    def generate(self, max_length=200, seed_text=None):
        """Just let the LM waffle a little bit"""

        if seed_text != None:
            random.seed(seed_text)
        
        output = ["^", "^"]   # our initialisation scheme
        
        for i in range(max_length):
            #get the dist of next chars
            prefix = "".join(output[i:i+2])
            #print(prefix)
            cand_next_chars = list(self.counts[prefix].keys())
            next_char_counts = list(self.counts[prefix].values())

            next_char = random.choices(cand_next_chars, weights=next_char_counts)[0]

            if next_char == "$":
                break
            else:
                output.append(next_char)
        return "".join(output[2:])


if __name__ == "__main__":
    with open("../data/raw/train.en.txt") as f:
        text = f.read()

    sentences = normalise_text(text)

    #print(calculate_counts(sentences))
    trigram_model = TrigramModel()

    trigram_model.fit(sentences)

    perplexity = trigram_model.perplexity(sentences)

    gen_text = trigram_model.generate(seed_text=42)

    print(f"Model Perplexity: {perplexity}")
    print(gen_text)

    