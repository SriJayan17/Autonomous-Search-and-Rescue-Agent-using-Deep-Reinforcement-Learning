from Project.FrontEnd.RealTimeEnvironment import RealTimeEnvironment
from Project.FrontEnd.StaticEnvironment import StaticEnvironment
from Project.FrontEnd.DynamicEnvironment import DynamicEnvironment
from Project.FrontEnd.RealTimeEnvironment import RealTimeEnvironment

class Environment:

    def __init__(self, type = 'static'):

        if type.lower() == 'static':
            StaticEnvironment()
        
        elif type.lower() == 'test':
            RealTimeEnvironment()

        elif type.lower() == 'dynamic':
            DynamicEnvironment()