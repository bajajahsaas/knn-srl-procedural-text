import pickle
import sys
import random
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

random.shuffle(data)
trainsplit = int(0.75*len(data))
with open('train.pkl', 'wb') as f:
    pickle.dump(data[:trainsplit], f)
with open('val.pkl', 'wb') as f:
    pickle.dump(data[trainsplit:], f)

