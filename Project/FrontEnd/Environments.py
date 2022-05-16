from Project.FrontEnd.StaticEnvironment import StaticEnvironment
from Project.FrontEnd.DynamicEnvironment import DynamicEnvironment

class Environment:

    def __init__(self, type = 'static'):

        if type.lower() == 'static':
            StaticEnvironment()
        else:
            DynamicEnvironment()