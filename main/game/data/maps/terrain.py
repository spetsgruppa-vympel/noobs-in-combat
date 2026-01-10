"""
Terrain type definitions for Snakes in Combat.

Defines all terrain types with their gameplay properties including movement costs,
defensive bonuses, and special characteristics. Terrain affects unit movement,
combat effectiveness, and line of sight.
"""


class terrain:
    """
    Terrain type with gameplay properties.

    Each terrain type affects gameplay through movement costs, defensive bonuses,
    and visibility blocking. These properties are used by the game engine to
    calculate movement ranges, combat outcomes, and line of sight.

    Attributes:
        name: Display name of the terrain type
        movementCost: Base movement cost multiplier (1.0 = normal)
        blocksSight: Whether this terrain blocks line of sight
        intDefenseBonus: Flat damage reduction bonus (integer armor points)
        prcntDefenseBonus: Percentage damage reduction (0-100)
        perks: List of special terrain abilities (future feature)
        icon: Visual representation identifier for rendering
    """

    def __init__(
            self,
            name,
            movementCost,
            blocksSight,
            intDefenseBonus,
            prcntDefenseBonus,
            perks=None,
            icon=None
    ):
        """
        Initialize a terrain type with gameplay properties.

        Args:
            name: Display name string
            movementCost: Movement cost multiplier (1.0 = normal speed)
            blocksSight: Boolean, True if terrain blocks line of sight
            intDefenseBonus: Integer armor bonus for units on this terrain
            prcntDefenseBonus: Percentage damage reduction (0-100)
            perks: Optional list of special terrain abilities
            icon: Optional icon identifier for rendering
        """
        self.name = name
        self.movementCost = movementCost
        self.blocksSight = blocksSight
        self.intDefenseBonus = intDefenseBonus
        self.prcntDefenseBonus = prcntDefenseBonus
        self.perks = perks if perks is not None else []
        self.icon = icon

    def __repr__(self):
        """String representation for debugging."""
        return (
            f"terrain(name='{self.name}', movementCost={self.movementCost}, "
            f"blocksSight={self.blocksSight}, intDefense={self.intDefenseBonus}, "
            f"prcntDefense={self.prcntDefenseBonus}%)"
        )


# Terrain type instances used throughout the game

# Plains: Default open terrain with no special properties
plains = terrain(
    name="Plains",
    movementCost=1.0,
    blocksSight=False,
    intDefenseBonus=0,
    prcntDefenseBonus=0
)

# Forest: Provides cover and slows movement
forest = terrain(
    name="Forest",
    movementCost=1.5,
    blocksSight=True,
    intDefenseBonus=0,
    prcntDefenseBonus=20
)

# Urban: City terrain with strong defensive position
urban = terrain(
    name="Urban",
    movementCost=1.25,
    blocksSight=True,
    intDefenseBonus=5,
    prcntDefenseBonus=20
)

# Mountains: Difficult terrain that blocks sight and provides strong defense
mountains = terrain(
    name="Mountains",
    movementCost=2.0,
    blocksSight=True,
    intDefenseBonus=5,
    prcntDefenseBonus=20
)

# Road: Fast movement corridor with defensive penalty
road = terrain(
    name="Road",
    movementCost=0.75,
    blocksSight=False,
    intDefenseBonus=0,
    prcntDefenseBonus=-10  # Negative bonus = penalty
)

# Highway: Very fast movement with greater defensive penalty
highway = terrain(
    name="Highway",
    movementCost=0.5,
    blocksSight=False,
    intDefenseBonus=0,
    prcntDefenseBonus=-20
)

# Debris: Light cover from rubble or destroyed structures
debris = terrain(
    name="Debris",
    movementCost=1.0,
    blocksSight=False,
    intDefenseBonus=2,
    prcntDefenseBonus=10
)