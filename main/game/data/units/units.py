class units:  # contains the attributes common to every single unit

    # define attributes
    def __init__(self, name, hp, cost, armor, sight, mobility, mobilityType, unitClass, heads, transport_weight, perks, weapons, description=None, icon=None):
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

        # the amount of characters in this unit (for example grunts have 3, sniper team have 2 etc)
        self.heads = heads

        # the amount of transport slots this unit takes
        self.transport_weight = transport_weight

        # perks, used as list
        self.perks = perks

        # weapons the unit possesses
        self.weapons = weapons

        # description of the unit
        self.description = description

        # icon of the unit
        self.icon = icon

def defineUnits():
    # imports
    from .unitData.mobilityTypes import footMobility, towedMobility, tireMobility, tracksMobility
    from .unitData.weapons import (
        assaultRifle, smg, grenade, machineGunLt, machineGunMd, sniperMd,
        antiTankRifle, bazooka, rpg, antiTankMd, machineGunVehicle, antiTankHEAT,
        cannonMdHE, atgm, autoCannonMdP, cannonHy, mortar, howitzer, fieldGun,
        fieldGunDirect, cannonMdP, machineGunPrimary, autoCannonLt, cannonLt,
        autoCannonMd, cannonMdHEArt, machineGunFast, cannonLtHE, cannonBT,
        cannonSP, resupply,
    )
    from .unitData.unitClasses import infantry, towed, vehicle
    from .perks import perks, radio, fuel, healing, transport, capture, multi_attack, suppressor, headshot

    # Define units and push them into the module’s global namespace
    g = globals()  # shorthand

    # Recon Tank
    g["Recon_Tank"] = units(
        name="Recon Tank",
        hp=50,
        cost=125,
        armor=17.5,
        sight=2,
        mobility=4,
        mobilityType=tracksMobility,
        unitClass=vehicle,
        heads=1,
        transport_weight=3,
        perks=[fuel.clone_with(modifier=10), radio.clone_with(modifier=1)],
        weapons=[cannonMdP, machineGunVehicle, atgm],
    )

    # Grunts
    g["Grunts"] = units(
        name="Grunts",
        hp=100,
        cost=30,
        armor=0,
        sight=2,
        mobility=2,
        mobilityType=footMobility,
        unitClass=infantry,
        heads=3,
        transport_weight=1,
        perks=[radio.clone_with(modifier=2), capture],
        weapons=[assaultRifle],
    )

    # Sniper Team
    g["Sniper_Team"] = units(
        name="Sniper Team",
        hp=50,
        cost=55,
        armor=0,
        sight=3,
        mobility=2,
        mobilityType=footMobility,
        unitClass=infantry,
        heads=2,
        transport_weight=1,
        perks=[radio.clone_with(modifier=1), headshot, capture],
        weapons=[sniperMd],
    )

    # Mobile Howitzer
    g["Mobile_Howitzer"] = units(
        name="Mobile Howitzer",
        hp=50,
        cost=140,
        armor=15,
        sight=1,
        mobility=3,
        mobilityType=tracksMobility,
        unitClass=towed,
        heads=1,
        transport_weight=3,
        perks=[fuel.clone_with(modifier=10)],
        weapons=[howitzer, machineGunVehicle],
    )

    # Light Machine Gunner
    g["Light_Machine_Gunner"] = units(
        name="Light Machine Gunner",
        hp=70,
        cost=35,
        armor=0,
        sight=2,
        mobility=2,
        mobilityType=footMobility,
        unitClass=infantry,
        heads=2,
        transport_weight=1,
        perks=[
            suppressor,
            capture,
            multi_attack.clone_with(modifier=1)
            if hasattr(multi_attack, "clone_with")
            else multi_attack,
        ],
        weapons=[machineGunLt],
    )

    # Medium Machine Gunner
    g["Medium_Machine_Gunner"] = units(
        name="Medium Machine Gunner",
        hp=100,
        cost=40,
        armor=0,
        sight=2,
        mobility=1,
        mobilityType=footMobility,
        unitClass=infantry,
        heads=2,
        transport_weight=2,
        perks=[
            suppressor,
            capture,
            multi_attack.clone_with(modifier=1)
            if hasattr(multi_attack, "clone_with")
            else multi_attack,
        ],
        weapons=[machineGunMd],
    )

    # Battle Tank
    g["Battle_Tank"] = units(
        name="Battle Tank",
        hp=60,
        cost=180,
        armor=27.5,
        sight=1,
        mobility=3,
        mobilityType=tracksMobility,
        unitClass=vehicle,
        heads=1,
        transport_weight=3,
        perks=[fuel.clone_with(modifier=10)],
        weapons=[cannonBT, machineGunVehicle],
    )

    # Supply Carrier
    g["Supply_Carrier"] = units(
        name="Supply Carrier",
        hp=60,
        cost=90,
        armor=12.5,
        sight=1,
        mobility=3,
        mobilityType=tracksMobility,
        unitClass=vehicle,
        heads=1,
        transport_weight=2,
        perks=[
            transport.clone_with(modifier=1),
            multi_attack.clone_with(modifier=1)
            if hasattr(multi_attack, "clone_with")
            else multi_attack,
        ],
        weapons=[resupply],
    )

    # Supply Truck
    g["Supply_Truck"] = units(
        name="Supply Truck",
        hp=60,
        cost=60,
        armor=5,
        sight=1,
        mobility=5,
        mobilityType=tireMobility,
        unitClass=vehicle,
        heads=2,
        transport_weight=2,
        perks=[transport.clone_with(modifier=1)],
        weapons=[resupply],
    )

    # Infantry Fighting Vehicle (IFV)
    g["IFV"] = units(
        name="Infantry Fighting Vehicle",
        hp=50,
        cost=130,
        armor=15,
        sight=1,
        mobility=4,
        mobilityType=tracksMobility,
        unitClass=vehicle,
        heads=1,
        transport_weight=3,
        perks=[fuel.clone_with(modifier=10), transport.clone_with(modifier=2)],
        weapons=[autoCannonMd, atgm, machineGunVehicle],
    )

    # SMG Squad
    g["SMG_Squad"] = units(
        name="SMG Squad",
        hp=90,
        cost=25,
        armor=0,
        sight=2,
        mobility=2,
        mobilityType=footMobility,
        unitClass=infantry,
        heads=3,
        transport_weight=1,
        perks=[
            multi_attack.clone_with(modifier=2)
            if hasattr(multi_attack, "clone_with")
            else multi_attack,
            capture,
        ],
        weapons=[smg, grenade],
    )

    # Grunts M
    g["Grunts_M"] = units(
        name="Grunts M",
        hp=100,
        cost=30,
        armor=0,
        sight=2,
        mobility=2,
        mobilityType=footMobility,
        unitClass=infantry,
        heads=3,
        transport_weight=1,
        perks=[healing.clone_with(modifier=0.15), capture],
        weapons=[assaultRifle],
    )

    # Armored Personnel Carrier (APC)
    g["APC"] = units(
        name="Armored Personnel Carrier",
        hp=50,
        cost=80,
        armor=12.5,
        sight=1,
        mobility=4,
        mobilityType=tracksMobility,
        unitClass=vehicle,
        heads=1,
        transport_weight=3,
        perks=[
            transport.clone_with(modifier=3),
            fuel.clone_with(modifier=10),
            radio.clone_with(modifier=2),
        ],
        weapons=[machineGunPrimary],
    )

    # TODO: recon car (stub)
    g["Recon_Car"] = units(
        name="Recon Car",
        hp=None,
        cost=None,
        armor=None,
        sight=None,
        mobility=None,
        mobilityType=None,
        unitClass=vehicle,
        heads=None,
        transport_weight=None,
        perks=None,
        weapons=None,
    )

    # TODO: Mortar
    g["Mortar"] = units(
        name="Mortar",
        hp=None,
        cost=None,
        armor=None,
        sight=None,
        mobility=None,
        mobilityType=towedMobility,
        unitClass=towed,
        heads=None,
        transport_weight=None,
        perks=None,
        weapons=[mortar],
    )

    # export list
    g["__all__"] = [
        "Recon_Tank",
        "Grunts",
        "Sniper_Team",
        "Mobile_Howitzer",
        "Light_Machine_Gunner",
        "Medium_Machine_Gunner",
        "Battle_Tank",
        "Supply_Carrier",
        "Supply_Truck",
        "IFV",
        "SMG_Squad",
        "Grunts_M",
        "APC",
        "Recon_Car",
        "Mortar",
    ]


# trigger once at import so globals exist
defineUnits()


# TODO: all units as objects after finishing related stuff