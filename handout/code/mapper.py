#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys
from collections import defaultdict


NUM_OF_HASH_FUNCTIONS = 100

if __name__ == "__main__":
    # VERY IMPORTANT:
    # Make sure that each machine is using the
    # same seed when generating random numbers for the hash functions.
    np.random.seed(seed=42)

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

        # update list (set) with all shingles
        all_shingles = np.union1d(shingles, all_shingles)

    # create hash functions
    hash_functions = []
    for i in range(NUM_OF_HASH_FUNCTIONS):
        hash_functions.append(lambda x: x ^ np.random.randint(1)) # TODO define proper hashing functions

    # MinHashing
    ## init to inf (None)
    init_row = [None for i in range(len(shingles_in_video))]
    signature_matrix = [init_row for j in range(len(all_shingles))]

    # for each column c
    for c, video_shingles in enumerate(shingles_in_video.iteritems()):
        video = video_shingles[0]
        shingles_in_video = video_shingles[1]
        # for each row r
        for shingle in all_shingles:
            # if c has 1 in row r
            if shingle in shingles_in_video:
                # for each hash function h_i do
                for i, hash_function in enumerate(hash_functions):
                    if signature_matrix[i][c] == None:
                        signature_matrix[i][c] = hash_function(shingle)
                    else:
                        signature_matrix[i][c] = min(signature_matrix[i][c], hash_function(shingle))
    print signature_matrix
