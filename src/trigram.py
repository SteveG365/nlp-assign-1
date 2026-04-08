
import numpy as np
import random

def calculate_trigram_counts(train_sentences):
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
                    counts[prefix][char] = 1    # init
            else:
                counts[prefix] = {}             # init
                counts[prefix][char] = 1
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

def calculate_continuation_counts(train_sentences):
    """
    continuation_counts[char] = number of unique previous chars that precede char

    For Kneser-Ney lower-order distribution:
        P_cont(char) = continuation_counts[char] / total_unique_bigrams
    """
    previous_sets = {}

    for sentence in train_sentences:
        for i in range(len(sentence) - 1):
            prev_char = sentence[i]
            char = sentence[i+1]

            if char not in previous_sets:
                previous_sets[char] = set()

            previous_sets[char].add(prev_char)

    continuation_counts = {}
    for char in previous_sets:
        continuation_counts[char] = len(previous_sets[char])

    total_unique_bigrams = sum(continuation_counts.values())

    return continuation_counts, total_unique_bigrams


def calculate_bigram_counts(train_sentences):
    """
    bigram_counts[prev_char][char] = number of times char follows prev_char
    """
    bigram_counts = {}

    for sentence in train_sentences:
        for i in range(len(sentence) - 1):
            prev_char = sentence[i]
            char = sentence[i+1]

            if prev_char not in bigram_counts:
                bigram_counts[prev_char] = {}

            bigram_counts[prev_char][char] = bigram_counts[prev_char].get(char, 0) + 1

    return bigram_counts

class TrigramModel:


    def fit(self, train_sentences):
        """Estimate trigram counts/probabilities from training texts."""
        
        self.counts = calculate_trigram_counts(train_sentences)
        self.history_counts = calculate_history_counts(self.counts)

        self.bigram_counts = {}
        self.continuation_counts = {}
        self.total_unique_bigrams = 0
        

    def get_probability(self, prefix, char, D = 0.75):
        """
        Fetches the relevant probability p(char | prefix) and -
        uses Kneser Ney smoothing to steal from the rich and give to the poor
        """

        count = self.counts.get(prefix, {}).get(char, 0)
        total = self.history_counts.get(prefix, 0)

        if total > 0:
            unique_continuations = len(self.counts[prefix])
            lambda_weight = (D * unique_continuations) / total
            trigram_part = max(count - D, 0) / total
        else:
            trigram_part = 0.0
            lambda_weight = 1.0

        continuation_count = self.continuation_counts.get(char, 0)

        if self.total_unique_bigrams > 0:
            lower_order_prob = continuation_count / self.total_unique_bigrams
        else:
            lower_order_prob = 1e-12

        prob = trigram_part + lambda_weight * lower_order_prob
        
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




    