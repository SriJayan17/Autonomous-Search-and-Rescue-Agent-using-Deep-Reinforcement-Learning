# import matplotlib.pyplot as plt

# sample = list(range(0,1000,2))
# plt.plot(sample)
# plt.xlabel('Time')
# plt.ylabel('Sample')
# plt.show()

import pickle

# target = [1,2,3,4],
# pickle.dump(target, open('log.txt','wb'))
try:
    target = pickle.load(open('log.txt','rb'))[0]
    print(target)
    print(type(target))
except EOFError as e:
    print('File empty bro!')