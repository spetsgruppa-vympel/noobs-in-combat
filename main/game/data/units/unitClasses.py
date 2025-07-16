# contains weapons, mobility types and unit classes


class unitClasses:  # contains the classes of units, NOT TO BE CONFUSED WITH THE FILE units have their own classes
    # this classes refers to the unit classes, not the python classes

    # define attributes
    def __init__(self, name):
        # name of the unit
        self.name = name


# define objects for unitClasses
# infantry units
infantry = unitClasses("Infantry")
# towed units
towed = unitClasses("Towed")
# vehicles
vehicle = unitClasses("Vehicle")
# ships
ship = unitClasses("Ship")


# TODO: if needed, improve on the class with required attributes

class mobilityTypes:  # contains the mobility types of units

    # define attributes
    def __init__(self, plainsCost, roadCost, highwayCost, forestCost, urbanCost, debrisCost, entersUrbanTerrain,
                 entersFortifications, isSea):
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


# define objects for mobilityTypes

# for infantry units
footMobility = mobilityTypes(0, 0.25, 0.5, -0.5, 1, 0, True, True, False)

# for towed units (artillery, towed anti tank guns, etc)
towedMobility = mobilityTypes(0, 0.25, 0.5, -0.5, 1, 0,False, True, False)

# for vehicles with tires
tireMobility = mobilityTypes(1, 0, 0, 1, 100, 0.5, False, False, False)

# for tracked vehicles
tracksMobility = mobilityTypes(0, 0, 0, 1, 100, 0.5,False, False, False)

# for halftrack vehicles
halftrackMobility = mobilityTypes(0.125, 0, 0, 1, 100, 0.5,False, False, False)

# for light ships
shiplightMobility = mobilityTypes(3333, 3333, 3333, 3333, 3333, 3333,False, False, True)

# for heavy ships
shipheavyMobility = mobilityTypes(3333, 3333, 3333, 3333, 3333, 3333,False, False, True)


#TODO: need to properly implement sea units (can be delayed, not necessary)


class weapons:  # define weapon class for units to use
    def __init__(self, name, damage, times, ap, ammo, suppression, cooldown, fireRange, canAttack, indirect,
                 perks=None, description=None, icon=None):
        # name of the weapon in string
        self.name = name

        # baseline damage of the weapon
        self.damage = damage

        # how many times the weapon attacks
        self.times = times

        # ability to ignore armor of the weapon
        self.ap = ap

        # how many times you can fire before requiring a resupply
        self.ammo = ammo

        # suppression effect of the weapon for infantry
        self.suppression = suppression

        # firing cooldown in turns, 0 means you can fire again next turn
        self.cooldown = cooldown

        # firing range in tiles
        self.fireRange = fireRange

        # list of the unitClasses the unit can attack
        self.canAttack = canAttack

        # is the attack indirect?
        self.indirect = indirect

        # if the weapon has any special ability
        self.perks = perks
        # TODO: weapon ability classes

        # description of the weapon
        self.description = description

        # icon of the weapon
        self.icon = icon

    def __str__(self):
        return f"{self.name} (DMG: {self.damage} x{self.times}, AP: {self.ap}, Ammo: {self.ammo}, Range: {self.fireRange}, Indirect: {self.indirect})"

    def __repr__(self):
        return (
            f"weapons(name={self.name!r}, damage={self.damage}, times={self.times}, "
            f"ap={self.ap}, ammo={self.ammo}, suppression={self.suppression}, "
            f"cooldown={self.cooldown}, fireRange={self.fireRange}, "
            f"canAttack={self.canAttack}, indirect={self.indirect}, "
            f"abilities={self.abilities}, description={self.description}, icon={self.icon})"
        )


# set objects of weapons
# "P" at the end of a weapon name means "plus"
boltRifle = weapons("Rifle", 17, 2, 0, 6, 3, 0, 1, [infantry, towed, vehicle], False)
smg = weapons("SMG", 9, 4, 0, 4, 3, 0, 1, [infantry, towed, vehicle], False)
grenade = weapons("Grenade", 35, 1, 0, 2, 2, 2, 1, [infantry, towed, vehicle], False)
assaultRifle = weapons("Assault Rifle", 15, 3, 0, 4, 3, 0, 1, [infantry, towed, vehicle], False)
machineGunLt = weapons("Light Machine Gun", 16, 4, 0, 3, 6, 1, 2, [infantry, towed, vehicle], False)
machineGunMd = weapons("Medium Machine Gun", 16, 4, 0, 5, 6, 1, 2, [infantry, towed, vehicle], False)
sniperLt = weapons("Light Sniper", 90, 1, 0, 5, 2, 0, 3, [infantry, towed, vehicle], False)
sniperMd = weapons("Heavy Sniper", 90, 1, 5, 5, 2, 0, 3, [infantry, towed, vehicle], False)
antiTankRifle = weapons("Anti-Tank Rifle", 25, 1, 15, 4, 2, 0, 2, [infantry, towed, vehicle], False)
bazooka = weapons("Bazooka", 40, 1, 12.5, 3, 2, 0, 1, [infantry, towed, vehicle], False)
rpg = weapons("RPG", 45, 1, 15, 3, 2, 0, 1, [infantry, towed, vehicle], False)
antiTankMd = weapons("Anti Tank Gun", 40, 1, 15, 3, 2, 0, 3, [infantry, towed, vehicle], False)
machineGunVehicle = weapons("Machine Gun", 15, 3, 3, 4, 3, 0, 1, [infantry, towed, vehicle], False)
antiTankHEAT = weapons("Anti Tank Gun (HEAT)", 50, 3, 15, 2, 2, 0, 3, [infantry, towed, vehicle], False)
cannonMdHE = weapons("Medium Cannon (HEAT)", 50, 1, 5, 2, 2, 0, 3, [infantry, towed, vehicle], False)
atgm = weapons("Anti-Tank Guided Missile", 50, 1, 20, 3, 2, 0, 4, [infantry, towed, vehicle], False)
autoCannonMdP = weapons("Medium Auto-Cannon", 24, 4, 10, 4, 4, 0, 2, [infantry, towed, vehicle], False)
cannonHy = weapons("Heavy Cannon", 45, 1, 15, 3, 2, 0, 3, [infantry, towed, vehicle], False)
mortar = weapons("Mortar", 19, 2, 0, 3, 3, 0, 4, [infantry, towed, vehicle], True)
howitzer = weapons("Howitzer", 45, 1, 5, 4, 3, 1, 8, [infantry, towed, vehicle], True)
fieldGun = weapons("Field Gun", 40, 1, 5, 4, 3, 0, 6, [infantry, towed, vehicle], True)
fieldGunDirect = weapons("Direct Fire", 40, 1, 12.5, 4, 2, 0, 3, [infantry, towed, vehicle], False)
cannonMdHECase = weapons("Indirect Cannon (HE)", 50, 1, 5, 3, 2, 0, 5, [infantry, towed, vehicle], True)
cannonHyHECase = weapons("Indirect Cannon (HE)", 60, 1, 7.5, 3, 2, 1, 5, [infantry, towed, vehicle], True)
rockets = weapons("Rockets", 23, 4, 5, 2, 3, 3, 6, [infantry, towed, vehicle], True)
resupply = weapons("Supply", 40, 1, 5, 6, 2, 0, 1, [infantry, towed, vehicle], 3)
cannonMdP = weapons("Medium Cannon", 40, 1, 15, 3, 2, 0, 2, [infantry, towed, vehicle], False)
machineGunPrimary = weapons("Machine Gun", 15, 3, 3, 4, 3, 0, 2, [infantry, towed, vehicle], False)
autoCannonLt = weapons("Light Auto-Cannon", 21, 3, 5, 4, 4, 0, 2, [infantry, towed, vehicle], False)
cannonLt = weapons("Light Cannon", 30, 1, 12.5, 4, 2, 0, 2, [infantry, towed, vehicle], False)
autoCannonMd = weapons("Medium Auto-Cannon", 24, 3, 10, 4, 4, 0, 2, [infantry, towed, vehicle], False)
cannonMdHEArt = weapons("Medium Indirect Cannon", 50, 1, 5, 3, 2, 0, 4, [infantry, towed, vehicle], True)
machineGunFast = weapons("Machine Gun", 15, 4, 3, 4, 4, 0, 2, [infantry, towed, vehicle], False)
cannontLtHE = weapons("Light Cannon (Indirect)", 45, 1, 2.5, 3, 2, 0, 2, [infantry, towed, vehicle], True)
cannonBT = weapons("Battle Tank Cannon", 50, 1, 15, 3, 2, 0, 4, [infantry, towed, vehicle], False)
cannonSP = weapons("Heavy BT Cannon", 70, 1, 20, 3, 2, 1, 4, [infantry, towed, vehicle], False)

# TODO: descriptions and icons, special abilities for grenades, a way to add ambush attribute
