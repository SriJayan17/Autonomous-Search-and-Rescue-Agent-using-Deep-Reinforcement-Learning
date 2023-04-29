import pickle

test = [-15,-15,-10,15,14,-12,50]
with open('Rescue_plan/path_trace.pickle','rb') as infile:
    test = pickle.load(infile)
    # print('Saved the rescue path')
    print(test)