
class units:  # contains the attributes common to every single unit

    # define attributes
    def __init__(self, name, hp, cost, armor, sight, mobility, mobilityType, unitClass, perks=None, weapons=None, description=None, icon=None):
        # name of the unit
        self.name = name

        # hp of the unit
        self.hp = hp

        # tix cost of the unit
        self.cost = cost

        # armor of the unit
        self.armor = armor

        # visibility of the unit in tiles
        self.sight = sight

        # mobility of the unit. mobility is calculated in cost, not tiles, affected by
        # mobilityType and terrainType of the tiles
        self.mobility = mobility

        # mobility type
        self.mobilityType = mobilityType

        # unit class from unitClass
        self.unitclass = unitClass

        # perks, used as list
        self.perks = perks

        # weapons the unit possesses
        self.weapons = weapons

        # description of the unit
        self.description = description

        # icon of the unit
        self.icon = icon

# TODO: all units as objects after finishing related stuff