"""
Perk and ability definitions for units and weapons.

Perks provide special abilities that modify unit behavior beyond basic stats.
They fall into four categories:

1. MODIFIER: Passive bonuses conditional on circumstances
   Example: +50% damage to urban units

2. REQUIREMENT: Things needed to perform actions
   Example: Radio required to call artillery

3. TRIGGERED: Actions that activate on specific events
   Example: Ambush bonus on first attack

4. PASSIVE: Always-active abilities
   Example: Cloaking

IMPORTANT: When using perks, ALWAYS use clone_with() to create instances
with specific values. Never use the base perk objects directly, as they
contain placeholder values (needs_assignment) that must be replaced.
"""


# ============================================================================
# ABILITY TYPE DEFINITIONS
# ============================================================================

class class_ability_type:
    """
    Ability type classification.

    Used to categorize perks by their behavior pattern, which helps
    the game engine know when and how to apply them.
    """

    def __init__(self, name):
        """
        Initialize an ability type.

        Args:
            name (str): Type identifier
        """
        self.name = name


# Define the four ability type categories
requirement_type = class_ability_type("requirement")
modifier_type = class_ability_type("modifier")
triggered_type = class_ability_type("triggered")
passive_type = class_ability_type("passive")


# ============================================================================
# PERK CLASS
# ============================================================================

class perks:
    """
    Special ability that modifies unit/weapon behavior.

    Perks are flexible ability containers that can represent various
    gameplay mechanics. They use a factory pattern (clone_with) to
    create specific instances from templates.

    Attributes:
        name (str): Ability name
        abilityType (class_ability_type): Category (modifier/requirement/etc)
        modifier (varies): Effect value (damage mult, range, etc)
        condition (varies): When this ability applies
        requires (list): Prerequisites for this ability
        triggerEvent (str): Event that activates this ability
        duration (varies): How long effect lasts
        description (str): UI description text
        icon (str): Icon identifier for rendering
    """

    def __init__(
            self,
            name,
            abilityType,
            modifier=None,
            condition=None,
            requires=None,
            triggerEvent=None,
            duration=None,
            description=None,
            icon=None
    ):
        """
        Initialize a perk.

        Args:
            name (str): Perk name
            abilityType (class_ability_type): Type category
            modifier: Value or multiplier for effect
            condition: When this perk applies
            requires (list): Prerequisites
            triggerEvent (str): Activation event
            duration: Effect duration
            description (str): UI text
            icon (str): Icon identifier
        """
        self.name = name
        self.abilityType = abilityType
        self.modifier = modifier
        self.condition = condition
        self.requires = requires
        self.triggerEvent = triggerEvent
        self.duration = duration
        self.description = description
        self.icon = icon

    def clone_with(
            self,
            name=None,
            abilityType=None,
            modifier=None,
            condition=None,
            requires=None,
            triggerEvent=None,
            duration=None
    ):
        """
        Create a copy of this perk with modified attributes.

        This is the PRIMARY way to use perks. Base perks are templates with
        needs_assignment placeholders. Use clone_with() to create actual
        instances with real values.

        Args:
            name (str, optional): Override name
            abilityType (class_ability_type, optional): Override type
            modifier (optional): Set modifier value
            condition (optional): Set condition
            requires (list, optional): Set requirements
            triggerEvent (str, optional): Set trigger
            duration (optional): Set duration

        Returns:
            perks: New perk instance with specified values

        Raises:
            ValueError: If any attribute is still needs_assignment

        Example:
            # Template defines radio as needing a modifier (range)
            radio = perks("Radio", requirement_type, needs_assignment, ...)

            # Create instance with specific range
            unit_radio = radio.clone_with(modifier=3)  # 3 tile range
        """
        # Determine final attribute values (use new value if provided, else keep existing)
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

        # Validate that no placeholders remain
        for key, value in attrs.items():
            if value == 'needs_assignment':
                raise ValueError(
                    f"Attribute '{key}' requires assignment in clone_with(). "
                    f"Cannot create perk with needs_assignment placeholder."
                )

        # Create and return new perk instance
        return perks(**attrs)


# Sentinel object for placeholder values
needs_assignment = object()

# ============================================================================
# PERK TEMPLATE DEFINITIONS
# ============================================================================
# These are templates. ALWAYS use clone_with() to create actual instances.
# ============================================================================

# Radio: Required to coordinate with other units or call support
# Modifier = radio range in tiles
radio = perks(
    name="Radio",
    abilityType=requirement_type,
    modifier=needs_assignment,  # Must specify range when cloning
    duration=needs_assignment  # Must specify duration when cloning
)

# Amphibious: Can move through water tiles
amphibious = perks(
    name="Amphibious",
    abilityType=passive_type
)

# Ambush: Bonus damage on surprise attacks
# Requires = list of requirements (like "undetected")
ambush = perks(
    name="Ambush",
    abilityType=triggered_type,
    requires=needs_assignment  # Must specify requirements when cloning
)

# Capture: Can capture buildings and objectives
capture = perks(
    name="Capture",
    abilityType=requirement_type
)

# Destruction: Bonus damage to structures
destruction = perks(
    name="Destruction",
    abilityType=modifier_type
)

# Fuel: Limited operational range (vehicles)
# Modifier = number of turns before refueling
fuel = perks(
    name="Fuel",
    abilityType=requirement_type,
    modifier=needs_assignment  # Must specify fuel capacity when cloning
)

# Headshot: Bonus damage to infantry (snipers)
headshot = perks(
    name="Sniper",
    abilityType=modifier_type
)

# Healing: Restores HP to nearby units (medics)
# Modifier = % HP healed per turn
healing = perks(
    name="Medic",
    abilityType=passive_type,
    modifier=needs_assignment  # Must specify heal rate when cloning
)

# Multi-attack: Can attack multiple times per turn
# Modifier = number of extra attacks allowed
multi_attack = perks(
    name="Multi-attack",
    abilityType=requirement_type,
    modifier=needs_assignment  # Must specify attack count when cloning
)

# Multi-move: Can move multiple times per turn
# Modifier = number of extra moves allowed
multi_move = perks(
    name="Multi-move",
    abilityType=requirement_type,
    modifier=needs_assignment  # Must specify move count when cloning
)

# No turret: Cannot fire while moving
no_turret = perks(
    name="No turret",
    abilityType=requirement_type
)

# Resistance: Damage reduction against specific types
# Modifier = % damage reduced
resistance = perks(
    name="Resistance",
    abilityType=modifier_type,
    modifier=needs_assignment  # Must specify reduction % when cloning
)

# Scouting: Extended vision range
# Modifier = extra sight range in tiles
scouting = perks(
    name="Scouting",
    abilityType=modifier_type,
    modifier=needs_assignment  # Must specify extra range when cloning
)

# Suppressor: Reduces suppression effect on this unit
suppressor = perks(
    name="Suppressor",
    abilityType=passive_type
)

# Transport: Can carry other units
# Modifier = transport capacity (weight units can carry)
transport = perks(
    name="Transport",
    abilityType=requirement_type,
    modifier=needs_assignment  # Must specify capacity when cloning
)

# ============================================================================
# TODO: FUTURE PERKS
# ============================================================================
# - Cloaked: Invisible until attacking or detected
# - Add descriptions and icons for all perks