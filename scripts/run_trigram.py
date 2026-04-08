import os
import sys
from src.trigram import TrigramModel
from src.normalisation import normalise_text
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trigram")
    
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    with open(args.file_name) as f:
        text = f.read()

    sentences = normalise_text(text)

    trigram_model = TrigramModel()

    trigram_model.fit(sentences)

    perplexity = trigram_model.perplexity(sentences)

    gen_text = trigram_model.generate(seed_text=42)

    print(f"Model Perplexity: {perplexity}")
    print(gen_text)