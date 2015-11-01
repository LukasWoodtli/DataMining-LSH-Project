#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

#fp = open(r"C:\Dropbox\___DM_Project2\weightvector_mapper.txt", "r")
fp = open(r"weightvector_mapper.txt", "r")

## reducer: each line is the resulting weight vector of a mapping step
##          -> goal: avarage weight vectors and write it to a file

DIMENSION = 400
w = np.array([0]*(DIMENSION+1))

for i, line in enumerate(fp):
#for line in sys.stdin:
    line = line.strip()
    wi = np.fromstring(line, sep=' ')
    w = w + wi

##Avarage
w_avarage = w/(i+1)
print(w_avarage)
np.savetxt('weightvector_reducer.txt', (w), newline=' ')
