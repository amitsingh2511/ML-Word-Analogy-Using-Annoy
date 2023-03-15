import argparse
import os
import sys
from word_analogy import WordAnalogy
import enchant

parser = argparse.ArgumentParser(description = 'Word Analogy', add_help=True)
parser.add_argument('PATH', metavar = 'path', type = str, help = 'Path to embedding file')
parser.add_argument('WORD1', metavar = 'word1', type = str, help = 'word 1')
parser.add_argument('WORD2', metavar = 'word2', type = str, help = 'word 2')
parser.add_argument('WORD3', metavar = 'word3', type = str, help = 'word 3')
args = parser.parse_args()
embedding_file = args.PATH
word1 = args.WORD1
word2 = args.WORD2
word3 = args.WORD3
d = enchant.Dict('en_US')

if os.path.isfile(embedding_file):
    embeddings = WordAnalogy.from_embeddings_file(embedding_file)
    if d.check(word1) and d.check(word2) and d.check(word3):
        print('{} : {} :: {} : {}'.format(word1, word2, word3, \
            embeddings.compute_and_print_analogy(word1, word2, word3)))
    else:
        print("Enter a valid English word!")
else:
    print("File path is not valid!")

