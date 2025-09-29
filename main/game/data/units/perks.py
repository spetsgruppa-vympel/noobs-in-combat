# contains perks used by buildings or units

class class_ability_type:  # define ability type class used l8r
    def __init__(self,
                 name):
        self.name = name

requirement_type = class_ability_type("requirement")
modifier_type = class_ability_type("modifier")
triggered_type = class_ability_type("triggered")
passive_type = class_ability_type("passive")

class perks:  # define perks class used by units and weapons
    def __init__(self,
                 name,  # name of the ability, also used to redirect to the adequate functions
                 abilityType,  # e.g. 'modifier', 'requirement', 'triggered', 'passive'
                 modifier=None,  # e.g. damageMult 0.5, specific use cases will be specified in the objects
                 condition=None,  # e.g. terrain = urban
                 requires=None,  # e.g. ['radio']
                 triggerEvent=None,  # e.g. 'ambush'
                 duration=None,  # e.g. 1 turn, 'until_used', etc.
                 description=None,  # description
                 icon=None  # icon

                 # modifier means a passive damage modifier conditioned by something e.g. +50% damage to urban units for flamethrowers
                 # requirement means the unit/attack requires something in order to execute an action, e.g. radio
                 # triggered means that on a certain action (trigger) it does something, e.g. ambush
                 # passive means a passive ability that always applies to the unit with exceptions e.g. cloaked

                 ):
        self.name = name
        self.abilityType = abilityType
        self.modifier = modifier
        self.condition = condition
        self.requires = requires
        self.triggerEvent = triggerEvent
        self.duration = duration
        self.description = description
        self.icon = icon

        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH
        # when using a perk ALWAYS USE CLONE_WITH


    def clone_with(self,  # function to clone an object with different attributes
                   name=None,
                   abilityType=None,
                   modifier=None,
                   condition=None,
                   requires=None,
                   triggerEvent=None,
                   duration=None,
                   ):
        # prepare new attribute values, fallback to existing ones if none are specified
        attrs = {
            'name': name if name is not None else self.name,
            'abilityType': abilityType if abilityType is not None else self.abilityType,
            'modifier': modifier if modifier is not None else self.modifier,
            'condition': condition if condition is not None else self.condition,
            'requires': requires if requires is not None else self.requires,
            'triggerEvent': triggerEvent if triggerEvent is not None else self.triggerEvent,
            'duration': duration if duration is not None else self.duration,
            'description': self.description,
            'icon': self.icon,
        }

        # validate that none of the attributes are 'needs_assignment'
        for key, value in attrs.items():
            if value == 'needs_assignment':
                raise ValueError(f"attribute '{key}' requires assignment and cannot be 'needs_assignment'.")

        # create and return new instance with validated attributes
        return perks(**attrs)

needs_assignment = object()  # define needs_assignment used in clone_with and objects to delay assignment

# radio perk, uses modifier for range in tiles
radio = perks("Radio", requirement_type, needs_assignment, None,None,
              None, needs_assignment, None, None, )

amphibious = perks("Amphibious", passive_type,None, None,
                   None, None, None, None, )

ambush = perks("Ambush", triggered_type, None, None,
               needs_assignment, None, None, None)

capture = perks("Capture", requirement_type, None, None, None,
                None, None, None, None)

destruction = perks("Destruction", modifier_type, None, None, None,
                    None, None, None, None)

# uses modifier for amount of turns before needing to be resupplied
fuel = perks("Fuel", requirement_type, needs_assignment, None, None,
             None, None, None, None)

headshot = perks("Sniper", modifier_type, None, None,
                 None, None, None, None, None)

# uses modifier for %hp healed per turn
healing = perks("Medic",passive_type, needs_assignment, None,
                None, None, None, None, None)

# uses modifier for the amount of times the unit is allowed to attack this turn
multi_attack = perks("Multi-attack", requirement_type, needs_assignment, None,
                     None, None, None, None, None)

# ditto for movement
multi_move = perks("Multi-move", requirement_type, needs_assignment, None,
                   None, None, None, None, None)

no_turret = perks("No turret", requirement_type, None, None,
                  None, None, None, None, None)

# uses modifier for % damage reduced
resistance = perks("Resistance", modifier_type, needs_assignment, None,
               None, None, None, None, None)

# uses modifier for tile range increased
scouting = perks("Scouting", modifier_type, needs_assignment, None,
                 None, None, None, None, None)

suppressor = perks("Suppressor", passive_type, None, None,
                   None, None, None, None, None)

# uses modifier for accepted transport weight
transport = perks("Transport", requirement_type, needs_assignment, None,
                  None, None, None, None, None)

# TODO: descriptions and icons, the perks too
# TODO: cloaked, napalm, i dont want to do them