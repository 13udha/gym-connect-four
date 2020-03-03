import copy
import numpy as np

# https://codereview.stackexchange.com/questions/198092/create-a-probability-list-which-sums-up-to-one
def kahansum(input):
    summ = c = 0
    for num in input:
        y = num - c
        t = summ + y
        c = (t - summ) - y
        summ = t
    return summ

def probChoice(l, f=None, print_probabilities=False):
    """
    chooses an index from the list l according to its entry's probabilities.
    The factor f decides how high peaks are.
    If f=None, simply the highest or if multiple highest a random choice of those is returned.
    Entries with -1000 cannot be picked.
    print_probabilities is a debug print statement which shows the probabilities for the indices
    """
    probabilities = []
    mapping = {}
    for index, entry in enumerate(l):
        if entry != -1000:
            probabilities.append(entry)
            mapping[len(probabilities)-1] = index
    assert len(probabilities) == len(mapping), "Mapping and probabilities out of sync"

    min_entry = min(probabilities)
    max_entry = max(probabilities)
    if not f:
        possible_indices = np.argwhere(probabilities == np.amax(probabilities)).flatten()
        return mapping[np.random.choice(possible_indices)]
    # Shifting the values is not 100% correct - looses some of the distinguashablity of the values.
    probabilities = [ x + 1.0*abs(min_entry) + 0.1 for x in probabilities]

    factor = 1.1 / min(probabilities)
    probabilities = [x * factor for x in probabilities] # multiply to make all values a minimum of 1.1

    probabilities = [x ** f for x in probabilities] # accounting for f - take every entry to the power of f
    sum_of_probabilities = kahansum(probabilities)
    probabilities = [x/sum_of_probabilities for x in probabilities] # calculating a probability distribution

    if print_probabilities:
        prop_mapping = {}
        for i, prop in enumerate(probabilities):
            prop_mapping[mapping[i]] = prop
        print('probabilities: {}'.format(prop_mapping))

    choice = np.random.choice(range(len(probabilities)), p=probabilities)

    return mapping[choice]



if __name__ == '__main__':
    l1 = [0.2, -0.4,-1000, -0.34, -1000, -0.9, 0.9]
    l2 = [-1000, 0.2, 0.7, -0.4, -0.34, -0.9, 0.9]
    l3 = [0.2, 0.7, -0.4, -0.34, -0.9, 0.9, -1000]


    print("Choosing index {}".format(probChoice(l1, 5, True)))
    print("Choosing index {}".format(probChoice(l2)))
    print("Choosing index {}".format(probChoice(l3)))
