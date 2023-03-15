from FrontEnd.RealTimeEnvironment import RealTimeEnvironment
from FrontEnd.StaticEnvironment import StaticEnvironment
from FrontEnd.DynamicEnvironment import DynamicEnvironment
from FrontEnd.RealTimeEnvironment import RealTimeEnvironment

class Environment:

    def __init__(self, type = 'static'):

        if type.lower() == 'static':
            StaticEnvironment()
        
        elif type.lower() == 'test':
            RealTimeEnvironment()

        elif type.lower() == 'dynamic':
            DynamicEnvironment()