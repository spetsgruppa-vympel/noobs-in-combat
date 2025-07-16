class terrain:
    def __init__(self, name, movementCost, blocksSight, intDefenseBonus, prcntDefenseBonus, perks=None, icon=None):
        self.name = name  # name of the terrain type
        self.movementCost = movementCost  # base movement cost
        self.blocksSight = blocksSight  # does this block sight of units next to it?
        self.intDefenseBonus = intDefenseBonus  # provides additional "Armor" for units sitting in the tile
        self.prcntDefenseBonus = prcntDefenseBonus  # reduces incoming damage by x%
        self.perks = perks  # any additional perks the terrain has
        self.icon = icon  # icon of the terrain


# define objects for terrain class
plains = terrain("Plains", 1, False, 0, 0)
forest = terrain("Forest", 1.5, True, 0, 20)
urban = terrain("Urban", 1.25, True, 5, 20)
mountains = terrain("Mountains", 2, True, 5, 20)
road = terrain("Road", 0.75, False, 0, -10)
highway = terrain("Highway", 0.5, False, 0, -20)
debris = terrain("Debris", 1, False, 2, 10)