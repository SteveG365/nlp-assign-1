from collections import defaultdict


class BPETokenLearner:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges

    def fit(self, sentences):
        """Run iterative BPE merges on training data."""
        self.merge_hist = []
        # init tokenisation
        tok_sentences = []
        for s in sentences:
            tok_sentences.append(list(s))

        for _ in range(self.num_merges):
            # count the frequencies
            counts = self.get_pair_frequencies(tok_sentences)
            best_pair = max(counts)
            tok_sentences = self.merge_best_pair(tok_sentences, best_pair)

            # record merge history
            self.merge_hist.append(best_pair)

        self.tok_sentences = tok_sentences
    

    def get_pair_frequencies(self, tok_sentences):
        """Count adjacent symbol pair frequencies."""
        counts = defaultdict(int)

        for sent in tok_sentences:
            for i in range(len(sent) - 1):
                pair = (sent[i], sent[i + 1])
                counts[pair] += 1

        return counts

    def merge_best_pair(self, tok_sentences, best_pair):
        """
        Applies one merge iteration
        """
        merged_sentences = []

        a, b = best_pair
        merged_token = a + b

        for sentence in tok_sentences:
            new_sent = []
            i = 0

            while i < len(sentence):
                if i < len(sentence) - 1 and sentence[i] == a and sentence[i + 1] == b:
                    new_sent.append(merged_token)
                    i += 2
                else:
                    new_sent.append(sentence[i])
                    i += 1

            merged_sentences.append(new_sent)

        return merged_sentences

    def get_merge_history(self):
        """Return ordered list of merges."""
        return self.merge_hist

    def get_vocab(self):
        """Return final learned subword vocabulary."""
        V = set()
        for sentence in self.tok_sentences:
            for token in sentence:
                V.add(token)

        return V