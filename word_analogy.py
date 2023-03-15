from annoy import AnnoyIndex
import numpy as np

class WordAnalogy(object):
  def __init__(self, word_to_index, word_vectors):
    self.word_to_index = word_to_index
    self.word_vectors = word_vectors
    self.index_to_word = {v:k for k,v in self.word_to_index.items()}
    self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')
    for _, i in self.word_to_index.items():
      self.index.add_item(i, self.word_vectors[i])
    self.index.build(50)

  @classmethod
  def from_embeddings_file(cls, embedding_file):
    word_vectors = []
    word_to_index = {}
    print('Building...')
    with open(embedding_file) as fp:
      for line in fp.readlines():
        line = line.split(" ")
        word = line[0]
        vec = np.array([float(x) for x in line[1:]])
        word_to_index[word] = len(word_to_index)
        word_vectors.append(vec)
    return cls(word_to_index, word_vectors)
    
  def get_embedding(self, word):
    """
    Args:
      word (str)
    Returns:
      an embedding (numpy.ndarray)
    """
    return self.word_vectors[self.word_to_index[word]] 

  def get_closest_to_vector(self, vector, n=1):
    """Given a vector return it's n nearest neighbors
    Args:
      vector (np.ndarray): should match the size of the vectors in the Annoy index
      n (int): the number of neighbors to return 
    Returns:
      [str, str, ...]: words nearest to the vector, not ordered by distance
    """
    nn_indices = self.index.get_nns_by_vector(vector, n)
    return [self.index_to_word[neighbor] for neighbor in nn_indices]
    
  
  #derive analogy
  def compute_and_print_analogy(self, word1, word2, word3):
    """
    Print the solutions to analogies using word embeddings
    word1:word2::word3:____
    Args:
      word1 (str)
      word2 (str)
      word3 (str)
    """

    vec1 = self.get_embedding(word1)
    vec2 = self.get_embedding(word2)
    vec3 = self.get_embedding(word3)

    #hypothesis = word2 - word1 + word3
    difference = vec2 - vec1
    vec4 = difference + vec3

    closest_words = self.get_closest_to_vector(vec4, n=1)
    existing_words = set([word1, word2, word3])
    closest_words = [word for word in closest_words if word not in existing_words]

    if len(closest_words) == 0:
      print("Couldn't find any nearest neighbors for the vector")
      return
    
    
    return closest_words[0]


