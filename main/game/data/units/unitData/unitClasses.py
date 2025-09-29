# unitClasses.py

class unitClasses:
    # define attributes

    def __init__(self, name):
        # name of the unit class (string)
        self.name = name

# define objects for unitClasses (instances you can import and use)
# infantry units
infantry = unitClasses("Infantry")
# towed units
towed = unitClasses("Towed")
# vehicles
vehicle = unitClasses("Vehicle")
# ships
ship = unitClasses("Ship")
