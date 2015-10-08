#!/local/anaconda/bin/python
import numpy as np
import sys


NUMBER_OF_MINHASH_HASHFUNCTIONS = 100
ROWS_PER_BAND = 10


if __name__ == "__main__":

    # Make sure that each machine is using the same seed
    # Using NUMBER_OF_HASHFUNCTIONS Hashfunctions. Represented by a and b.
    np.random.seed(seed=42)
    minHashes = np.random.random_integers(0, 5000, size=(NUMBER_OF_MINHASH_HASHFUNCTIONS,2))
    
  
    for line in sys.stdin:
        line = line.strip()
        video_id = line[6:15]
        shingles = frozenset(np.fromstring(line[16:], sep=" "))

        # Create the signature
        # Use the hashfunctions as shown in II-47 to create signature of the file
        # But for only one column and instead of checking for a one in each row, 
        # we only calc for the shingles available in the list since its number
        # would equal its rownumber in the matrix.
        signature = np.full(NUMBER_OF_MINHASH_HASHFUNCTIONS, 20000)
        for shingle in shingles:
            for i,hash in enumerate(minHashes):
                hashed = (hash[0]*shingle + hash[1]) % 20000
                if hashed < signature[i]:
                    signature[i] = hashed

        # Emit the bands as keys sothat related files are sorted togther
        numBands = int(np.ceil(signature.size / ROWS_PER_BAND))
        for i in xrange (0, numBands):
             print( 
                # Key = Bandnumber + Signatur of Band
                str(i) + "_" + "".join(str(int(x)).zfill(5) for x in signature[i*ROWS_PER_BAND: (i+1)*ROWS_PER_BAND]) + "\t" 
                # Value = The shingles to filter in reducer + The VideoID
                + line[16:] + "_" + video_id
             ) 
