# ============================================================================
# unitClasses.py
# ============================================================================
"""
Unit class categorization system.

Defines broad categories of units (infantry, vehicles, ships, etc.) that
determine what weapons can target them and what terrain they can traverse.

Unit classes are used by:
- Weapon targeting restrictions (can this weapon attack this class?)
- Mobility rules (can this class enter urban terrain?)
- Balance calculations (ensuring diverse unit rosters)
"""


class unitClasses:
    """
    Unit category definition.

    Each unit belongs to exactly one class, which determines fundamental
    gameplay restrictions and interactions.

    Attributes:
        name (str): Class display name
    """

    def __init__(self, name):
        """
        Initialize a unit class.

        Args:
            name (str): Class name for display and identification
        """
        self.name = name


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

# Infantry: Foot soldiers (can capture, enter buildings)
infantry = unitClasses("Infantry")

# Towed: Artillery and equipment pulled by vehicles/infantry
towed = unitClasses("Towed")

# Vehicle: Motorized units (tanks, trucks, APCs)
vehicle = unitClasses("Vehicle")

# Ship: Naval units (future feature for water-based maps)
ship = unitClasses("Ship")

"""
Mobility type definitions for unit movement.

Each mobility type defines how units interact with different terrain:
- Movement costs (extra cost to enter terrain)
- Terrain restrictions (can/cannot enter certain tiles)
- Special movement properties (water traversal, building entry)

The base terrain cost is modified by these values to determine final
movement cost for each unit on each terrain type.
"""


class mobilityTypes:
    """
    Movement profile for a unit type.

    Defines terrain-specific movement costs and restrictions. These costs
    are ADDED to the base terrain cost to get final movement cost.

    Example:
        Plains base cost: 1.0
        Tracks mobility on plains: +0
        Final cost: 1.0

        Forest base cost: 1.5
        Tracks mobility on forest: +1.0
        Final cost: 2.5 (very slow!)

    Attributes:
        plainsCost (float): Extra cost for plains terrain
        roadCost (float): Extra cost for roads
        highwayCost (float): Extra cost for highways
        forestCost (float): Extra cost for forests
        urbanCost (float): Extra cost for urban areas
        debrisCost (float): Extra cost for debris
        entersUrbanTerrain (bool): Can enter urban tiles
        entersFortifications (bool): Can enter fortified positions
        isSea (bool): Can traverse water tiles
    """

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
        """
        Initialize a mobility type.

        Args:
            plainsCost (float): Extra movement cost on plains
            roadCost (float): Extra movement cost on roads
            highwayCost (float): Extra movement cost on highways
            forestCost (float): Extra movement cost in forests
            urbanCost (float): Extra movement cost in urban areas
            debrisCost (float): Extra movement cost in debris
            entersUrbanTerrain (bool): Whether unit can enter urban tiles
            entersFortifications (bool): Whether unit can enter fortifications
            isSea (bool): Whether unit can move on water
        """
        self.plainsCost = plainsCost
        self.roadCost = roadCost
        self.highwayCost = highwayCost
        self.forestCost = forestCost
        self.urbanCost = urbanCost
        self.debrisCost = debrisCost
        self.entersUrbanTerrain = entersUrbanTerrain
        self.entersFortifications = entersFortifications
        self.isSea = isSea


# ============================================================================
# MOBILITY TYPE DEFINITIONS
# ============================================================================

# Foot: Infantry and dismounted troops
# - No penalty on plains/roads
# - Bonus in forests (trained for it)
# - Can enter buildings and fortifications
footMobility = mobilityTypes(
    plainsCost=0,
    roadCost=0.25,  # Slight penalty (prefer cross-country)
    highwayCost=0.5,  # Larger penalty (highways too open)
    forestCost=-0.5,  # BONUS - infantry excel in forests
    urbanCost=1.0,  # Penalty for urban warfare
    debrisCost=0,
    entersUrbanTerrain=True,
    entersFortifications=True,
    isSea=False
)

# Towed: Artillery and heavy equipment pulled by hand/vehicle
# - Same as foot mobility but cannot enter buildings
towedMobility = mobilityTypes(
    plainsCost=0,
    roadCost=0.25,
    highwayCost=0.5,
    forestCost=-0.5,
    urbanCost=1.0,
    debrisCost=0,
    entersUrbanTerrain=False,  # Too bulky for buildings
    entersFortifications=True,
    isSea=False
)

# Tires: Wheeled vehicles (trucks, cars)
# - Excellent on roads
# - Terrible off-road (especially forests)
tireMobility = mobilityTypes(
    plainsCost=1.0,  # Penalty on rough ground
    roadCost=0,  # Roads are ideal
    highwayCost=0,  # Highways are ideal
    forestCost=1.0,  # Major penalty
    urbanCost=100,  # Effectively impassable (too narrow)
    debrisCost=0.5,  # Can navigate debris
    entersUrbanTerrain=False,
    entersFortifications=False,
    isSea=False
)

# Tracks: Tracked vehicles (tanks)
# - Good on plains and roads
# - Can handle rough terrain
# - Cannot enter urban areas (too large)
tracksMobility = mobilityTypes(
    plainsCost=0,  # Tracks excel on open ground
    roadCost=0,  # Also good on roads
    highwayCost=0,  # Excellent on highways
    forestCost=1.0,  # Penalty in forests
    urbanCost=100,  # Effectively impassable (too large)
    debrisCost=0.5,  # Can push through debris
    entersUrbanTerrain=False,
    entersFortifications=False,
    isSea=False
)

# Halftrack: Mixed wheel/track vehicles
# - Compromise between tires and tracks
# - Decent everywhere, excellent nowhere
halftrackMobility = mobilityTypes(
    plainsCost=0.125,  # Slight penalty
    roadCost=0,  # Good on roads
    highwayCost=0,  # Good on highways
    forestCost=1.0,  # Penalty in forests
    urbanCost=100,  # Cannot enter buildings
    debrisCost=0.5,  # Can handle debris
    entersUrbanTerrain=False,
    entersFortifications=False,
    isSea=False
)

# Ships (Light): Fast naval vessels
# - Can ONLY move on water
# - Effectively impassable on land (cost 3333)
shiplightMobility = mobilityTypes(
    plainsCost=3333,  # Impassable
    roadCost=3333,  # Impassable
    highwayCost=3333,  # Impassable
    forestCost=3333,  # Impassable
    urbanCost=3333,  # Impassable
    debrisCost=3333,  # Impassable
    entersUrbanTerrain=False,
    entersFortifications=False,
    isSea=True  # Can traverse water
)

# Ships (Heavy): Slow naval vessels with armor
# - Same restrictions as light ships but slower
shipheavyMobility = mobilityTypes(
    plainsCost=3333,  # Impassable
    roadCost=3333,  # Impassable
    highwayCost=3333,  # Impassable
    forestCost=3333,  # Impassable
    urbanCost=3333,  # Impassable
    debrisCost=3333,  # Impassable
    entersUrbanTerrain=False,
    entersFortifications=False,
    isSea=True  # Can traverse water
)