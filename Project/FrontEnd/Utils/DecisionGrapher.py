import matplotlib.pyplot as plt

class DecisionGrapher:
    
    def __init__(self):
        self.num_correct_decisions = []
        self.correct_dec_count = 0
        self.track = 0
    
    def correct_decision(self,is_correct):
        if is_correct: self.correct_dec_count += 1
        self.track += 1
        if self.track == 100:
            self.num_correct_decisions.append(self.correct_dec_count)
            # print(f'100 iters over, now num_correct_decisions: {self.num_correct_decisions}')
            self.correct_dec_count = 0
            self.track = 0
    
    def plot_decision_graph(self):
        # print(self.num_correct_decisions)
        print('Now plotting the decision graph')
        if len(self.num_correct_decisions) > 1:
            plt.plot(self.num_correct_decisions)
            plt.xlabel('Time')
            plt.ylabel('Number of correct decisions made')
            plt.show()