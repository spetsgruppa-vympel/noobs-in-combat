# unitClasses.py
# contains the classes of units, NOT to be confused with the file 'units' may have their own classes
# this file defines the unit classes used by weapons.canAttack lists etc.

class unitClasses:  # this class refers to the unit classes, not the python classes
    # define attributes
    __all__ = None

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

# export so you can do: from unit_classes import *
__all__ = ["unitClasses", "infantry", "towed", "vehicle", "ship"]
