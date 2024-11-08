class Optimization:
    def __init__(self):
        self.__variablecounter = -1

    @property
    def variableid(self):
        self.__variablecounter += 1
        return self.__variablecounter

    @property
    def variablecounter(self):
        return self.__variablecounter
