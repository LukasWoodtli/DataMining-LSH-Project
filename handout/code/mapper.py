#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys
from collections import defaultdict


NUM_OF_HASH_FUNCTIONS = 100
MAX_HASH_VAL = 2**32 - 1
ROWS_PER_BAND = 7

if __name__ == "__main__":
    # VERY IMPORTANT:
    # Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

    # create hash functions
    hash_functions = []
    for i in range(NUM_OF_HASH_FUNCTIONS):
        a = np.random.random_integers(0, MAX_HASH_VAL)
        b = np.random.random_integers(0, MAX_HASH_VAL)
        hash_functions.append(lambda x: (a * hash(x)  + b) % MAX_HASH_VAL)

    shingles_in_video = {}
    all_shingles = np.array([], dtype=int)

    for line in sys.stdin:
        line = line.strip()

        # get video id and shingles for video
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], dtype=int, sep=" ")
        # save shingles for each video
        assert video_id not in shingles_in_video, "Same video ID apeared twice!"
        shingles_in_video[video_id] = shingles

        minhashes = []
        for shingle in shingles: # TODO make sure every shingle apears only once
            minhash = None
            for hash_function in hash_functions:
                newMinhash = hash_function(shingle)
                if minhash == None or newMinhash < minhash:
                    minhash = newMinhash
            minhashes.append(minhash)

        lshashes = []
        for i in range(0, len(minhashes), ROWS_PER_BAND): # what to do with last band?
            band = minhashes[i:i + ROWS_PER_BAND]
            band = frozenset(band)
            lshashes.append(hash(band))

        print lshashes
