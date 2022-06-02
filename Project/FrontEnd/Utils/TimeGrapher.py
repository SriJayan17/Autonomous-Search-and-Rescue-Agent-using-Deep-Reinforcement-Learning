import pickle
import matplotlib.pyplot as plt

class TimeGrapher:
    
    def __init__(self,target_path):
        self.target_path = target_path
        self.time_lapse_record = []
        try:
            self.time_lapse_record = pickle.load(open(self.target_path,'rb'))
        except EOFError as e:
            print(f'Error occured: {e}')
            self.time_lapse_record = []
    
    def plot_graph(self,current_time_taken):
        self.time_lapse_record.append(current_time_taken)
        print(f'The time taken in the current iteration: {current_time_taken:.3f}s')
        
        if len(self.time_lapse_record) >= 2:
            plt.plot(self.time_lapse_record)
            plt.xlabel('No. of executions/training epochs')
            plt.ylabel('Time to reach victims(in seconds)')
            plt.show()
        
        print('Graph showed to the user')
        try:
            print('Inside the try block')
            pickle.dump(self.time_lapse_record,open(self.target_path,'wb'))
            print('Saved the time taken graph')
        except Exception as e:
            print(f'Error occured when storing in the log file: {e}')