# contains perks used by buildings or units

from enum import Enum, auto


class perks:  # define perks class used by units and weapons
    def __init__(self,
                 name,  # name of the ability, also used to redirect to the adequate functions
                 abilityType,  # e.g. 'modifier', 'requirement', 'trigger'
                 targetType=None,  # e.g. 'friendly', 'terrain', 'self', 'enemy', etc.
                 modifier=None,  # e.g. damageMult 0.5, specific use cases will be specified in the objects
                 condition=None,  # e.g. terrain = urban
                 requires=None,  # e.g. ['radio']
                 triggerEvent=None,  # e.g. 'ambush'
                 duration=None,  # e.g. 1 turn, 'until_used', etc.
                 description=None,  # description
                 icon=None

                 # modifier means a passive damage modifier conditioned by something e.g. +50% damage to urban units for flamethrowers
                 # requirement means the unit/attack requires something in order to execute an action, e.g. radio
                 # trigger means that on a certain action (trigger) it does something, e.g. ambush
                 ):
        self.name = name
        self.abilityType = abilityType
        self.targetType = targetType
        self.modifier = modifier
        self.condition = condition
        self.requires = requires
        self.triggerEvent = triggerEvent
        self.duration = duration
        self.description = description
        self.icon = icon

# radio perk, uses tar
radio = perks("radio", "requirement", None, 2, None, None, None, 1)

# TODO: descriptions and icons, the perks too
