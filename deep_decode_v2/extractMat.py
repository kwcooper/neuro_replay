

# script used to extract data from the mat file

import os
from scipy.io import loadmat

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

dataPath = '/Users/k/Desktop/fortin_lab/computedMats/Superchris_extraction_odor2s_PokeIn.mat'
#dataPath = os.path.join()
data_odor = loadmat(dataPath)
print(data_odor.keys())

'''
From epoch extraction SM:
'trialInfo' : Matrix containing information about the identity and
%           outcome of each trial. Rows represent individual trials;
%           columns are organized thusly:
%               Column 1) Trial performance. 1 = Correct, 0 = Incorrect
%               Column 2) Trial InSeq log. 1 = InSeq, 0 = OutSeq
%               Column 3) Trial Position.
%               Column 4) Trial Odor.
%               Column 5) Trial Number.
%               Column 6) Sequence Number.
%               Column 7) Poke Duration.
%               Column 8) Transposition Distance.
%               Column 9) Item Item Distance.


'''


print(data_odor['trialInfo'])
print('fin')