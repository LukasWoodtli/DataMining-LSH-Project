#!/local/anaconda/bin/python


import numpy as np
import sys


def print_duplicates(videos):
    """
    Prints all videos in the list once.
    Input:
    videos:     Unsorted list of videos, (doublings allowed)
    """
    result = []
    unique = np.unique(videos)
    for i in xrange(len(unique)):
        # When only one video, at this point no output at all.
        for j in xrange(i + 1, len(unique)):
            result.append( "%d\t%d" % (min(unique[i], unique[j]), #print 
                              max(unique[i], unique[j]))) 
    return result


def testFor90PercentSimilarity(shingleStringA, shingleStringB):
    shinglesA = set(np.fromstring(shingleStringA, sep=" "))
    shinglesB = set(np.fromstring(shingleStringB, sep=" "))
    similarity = float(len(shinglesA.intersection(shinglesB))) / float(len(shinglesA.union(shinglesB)))
    return similarity > 0.9


last_key = None
realDuplicates = dict()
result = []

for line in sys.stdin: 
    line = line.strip()
    key, value = line.split("\t")
    shingles, video_id = value.split("_")

    if last_key is None:
        last_key = key
    
    if key == last_key:
        # Key the same as for previous element
        for duplicateGroup in realDuplicates:
            if testFor90PercentSimilarity(duplicateGroup, shingles):
                realDuplicates[duplicateGroup].append(int(video_id))           
        realDuplicates[shingles] = [int(video_id)]
    else:
        # Key changed (previous line was k=x, this line is k=y)
        for duplicateGroup in realDuplicates:
            result += print_duplicates(realDuplicates[duplicateGroup])
        realDuplicates.clear()
        realDuplicates[shingles] = [int(video_id)]
        last_key = key

# also print the very last group
for shingles in realDuplicates:
    result += print_duplicates(realDuplicates[shingles])
    
# return all elements once by using set
for i in set(result):
    print i
