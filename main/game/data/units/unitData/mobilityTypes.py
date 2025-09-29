# mobility_types.py
# contains mobility types of units (movement cost modifiers and flags)
# comments kept lowercase and simple to match your style

class mobilityTypes:  # contains the mobility types of units
    # define attributes

    def __init__(
        self,
        plainsCost,
        roadCost,
        highwayCost,
        forestCost,
        urbanCost,
        debrisCost,
        entersUrbanTerrain,
        entersFortifications,
        isSea,
    ):
        # cost of movement added to plains for the respective unit
        self.plainsCost = plainsCost

        # ditto for road
        self.roadCost = roadCost

        # ditto for highway
        self.highwayCost = highwayCost

        # ditto for forest
        self.forestCost = forestCost

        # ditto for urban
        self.urbanCost = urbanCost

        # ditto for debris
        self.debrisCost = debrisCost

        # can the unit enter urban terrain?
        self.entersUrbanTerrain = entersUrbanTerrain

        # ditto for fortifications
        self.entersFortifications = entersFortifications

        # can the unit move on water tiles?
        self.isSea = isSea


# define objects for mobilityTypes (instances you can import and use)

# for infantry units
footMobility = mobilityTypes(0, 0.25, 0.5, -0.5, 1, 0, True, True, False)

# for towed units (artillery, towed anti tank guns, etc)
towedMobility = mobilityTypes(0, 0.25, 0.5, -0.5, 1, 0, False, True, False)

# for vehicles with tires
tireMobility = mobilityTypes(1, 0, 0, 1, 100, 0.5, False, False, False)

# for tracked vehicles
tracksMobility = mobilityTypes(0, 0, 0, 1, 100, 0.5, False, False, False)

# for halftrack vehicles
halftrackMobility = mobilityTypes(0.125, 0, 0, 1, 100, 0.5, False, False, False)

# for light ships
shiplightMobility = mobilityTypes(3333, 3333, 3333, 3333, 3333, 3333, False, False, True)

# for heavy ships
shipheavyMobility = mobilityTypes(3333, 3333, 3333, 3333, 3333, 3333, False, False, True)

# export so you can do: from mobility_types import *

