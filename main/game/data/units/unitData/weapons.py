# weapons.py
# contains the weapon class and all predefined weapon objects so you can just copy/paste this file
# comments follow your style (lowercase, simple, inline) and variable names kept as in your original code

# import unit class instances so canAttack lists work
from .unitClasses import infantry, towed, vehicle

# define weapon class (kept lowercase name 'weapons' to match your style)
class weapons:  # define weapon class for units to use
    __all__ = None

    def __init__(
        self,
        name,
        damage,
        times,
        ap,
        ammo,
        suppression,
        cooldown,
        fireRange,
        canAttack,
        indirect,
        perks=None,
        description=None,
        icon=None,
    ):
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
            f"description={self.description}, icon={self.icon})"
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
resupply = weapons("Supply", 40, 1, 5, 6, 2, 0, 1, [infantry, towed, vehicle], False)
cannonMdP = weapons("Medium Cannon", 40, 1, 15, 3, 2, 0, 2, [infantry, towed, vehicle], False)
machineGunPrimary = weapons("Machine Gun", 15, 3, 3, 4, 3, 0, 2, [infantry, towed, vehicle], False)
autoCannonLt = weapons("Light Auto-Cannon", 21, 3, 5, 4, 4, 0, 2, [infantry, towed, vehicle], False)
cannonLt = weapons("Light Cannon", 30, 1, 12.5, 4, 2, 0, 2, [infantry, towed, vehicle], False)
autoCannonMd = weapons("Medium Auto-Cannon", 24, 3, 10, 4, 4, 0, 2, [infantry, towed, vehicle], False)
cannonMdHEArt = weapons("Medium Indirect Cannon", 50, 1, 5, 3, 2, 0, 4, [infantry, towed, vehicle], True)
machineGunFast = weapons("Machine Gun", 15, 4, 3, 4, 4, 0, 2, [infantry, towed, vehicle], False)

# keep original variable name with its typo to avoid breaking existing code that references it
cannonLtHE = weapons("Light Cannon (Indirect)", 45, 1, 2.5, 3, 2, 0, 2, [infantry, towed, vehicle], True)
cannonBT = weapons("Battle Tank Cannon", 50, 1, 15, 3, 2, 0, 4, [infantry, towed, vehicle], False)
cannonSP = weapons("Heavy BT Cannon", 70, 1, 20, 3, 2, 1, 4, [infantry, towed, vehicle], False)


# export list so you can do: from weapons import *
__all__ = [
    "weapons",
    "boltRifle",
    "smg",
    "grenade",
    "assaultRifle",
    "machineGunLt",
    "machineGunMd",
    "sniperLt",
    "sniperMd",
    "antiTankRifle",
    "bazooka",
    "rpg",
    "antiTankMd",
    "machineGunVehicle",
    "antiTankHEAT",
    "cannonMdHE",
    "atgm",
    "autoCannonMdP",
    "cannonHy",
    "mortar",
    "howitzer",
    "fieldGun",
    "fieldGunDirect",
    "cannonMdHECase",
    "cannonHyHECase",
    "rockets",
    "resupply",
    "cannonMdP",
    "machineGunPrimary",
    "autoCannonLt",
    "cannonLt",
    "autoCannonMd",
    "cannonMdHEArt",
    "machineGunFast",
    "cannonLtHE",
    "cannonBT",
    "cannonSP",
]
