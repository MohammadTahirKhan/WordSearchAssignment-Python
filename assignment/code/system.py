"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""

from typing import List

import numpy as np
import scipy
from utils import utils
from utils.utils import Puzzle

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    Takes the raw feature vectors and reduces them down to the required number of
    dimensions. Note, the `model` dictionary is provided as an argument so that
    you can pass information from the training stage, e.g. if using a dimensionality
    reduction technique that requires training, e.g. PCA.

    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    
    # v = np.array(model['eigenvector'])
    # if len(v) == 0:
    # lab code
    covx = np.cov(data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N-60,N-1))
    v = np.fliplr(v)
        
    pca_data = np.dot((data - np.mean(data)), v)
    new_pca_data = np.dot(pca_data, v.transpose()) + np.mean(data)
    
    v = np.array(model['eigenvector'])
    # lab code
    if len(v) == 0:
        covx = np.cov(new_pca_data, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - N_DIMENSIONS, N - 1))
        v = np.fliplr(v)
        model['eigenvector'] = v.tolist()
    reconstruct_pca_data = np.dot((new_pca_data - np.mean(new_pca_data)), v)
    return reconstruct_pca_data
    

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is your classifier's training stage. You need to learn the model parameters
    from the training vectors and labels that are provided. The parameters of your
    trained model are then stored in the dictionary and returned. Note, the contents
    of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    The dummy implementation stores the labels and the dimensionally reduced training
    vectors. These are what you would need to store if using a non-parametric
    classifier such as a nearest neighbour or k-nearest neighbour classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels
    # e.g. Storing training data labels and feature vectors in the model.
    model = {}
    model["labels_train"] = labels_train.tolist()
    model['eigenvector'] = np.array([]).tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Dummy implementation of classify squares.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is the classification stage. You are passed a list of unlabelled feature
    vectors and the model parameters learn during the training stage. You need to
    classify each feature vector and return a list of labels.

    In the dummy implementation, the label 'E' is returned for every square.

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    x = np.dot(fvectors_test, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())
    nearest = np.argmax(dist, axis=1)
    
    return labels_train[nearest]
    
    

def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Dummy implementation of find_words.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This function searches for the words in the grid of classified letter labels.
    You are passed the letter labels as a 2-D array and a list of words to search for.
    You need to return a position for each word. The word position should be
    represented as tuples of the form (start_row, start_col, end_row, end_col).

    Note, the model dict that was learnt during training has also been passed to this
    function. Most simple implementations will not need to use this but it is provided
    in case you have ideas that need it.

    In the dummy implementation, the position (0, 0, 1, 1) is returned for every word.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    result = []
    for word in words:
        # wordFound = False
        word = word.upper()
        # for i in range(labels.shape[0]):
        #     for j in range(labels.shape[1]):
        #         if labels[i][j] == word[0]:
        #             for k in range(-1, 2):
        #                     for l in range(-1, 2):
        #                         if k == 0 and l == 0:
        #                             continue
        #                         if i>=labels.shape[0]-(len(word) - 1) and k == 1:
        #                             continue
        #                         if i<=len(word) - 1 and k == -1:
        #                             continue
        #                         if j >= labels.shape[1]-(len(word) - 1) and l == 1:
        #                             continue
        #                         if j<=len(word) - 1 and l == -1:
        #                             continue
        #                         if check_word(labels, word, i, j, k, l):
        #                             result += [(i,j, i + k * (len(word) - 1),j + l * (len(word) - 1))]
        #                             wordFound = True
        # if not wordFound:
        temp = []
        min = len(word)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if k == 0 and l == 0:
                            continue
                        if i>=labels.shape[0]-(len(word) - 1) and k == 1:
                            continue
                        if j >= labels.shape[1]-(len(word) - 1) and l == 1:
                            continue
                        unmatched = find_unmatching_letters(labels, word, i, j, k, l)
                        if unmatched != len(word):
                            if unmatched < min:
                                min = unmatched
                                temp = [(i,j, i + k * (len(word) - 1),j + l * (len(word) - 1))]
        result+=temp
    return result

def find_unmatching_letters(labels: np.ndarray, word: str, i: int, j: int, k: int, l: int) -> int:
    unmatched = 0
    for p in range(len(word)):
        if labels[(i + p * k)][(j + p * l)] != word[p] :
            unmatched += 1
    return unmatched