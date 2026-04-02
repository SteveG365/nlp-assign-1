import numpy as np
import argparse

def load_text(path):
    with open(path) as f:
        text = f.read()

    return text

if __name__ == "__main__":
    #import some example data
    with open("../data/raw/test.lid.txt") as f:
        text = f.read()
    print(text)
