"""Human-readable labels and a small data dictionary for the coded
categorical features. Source: ../Categorical_Feature_Codes.md
"""

CATEGORICAL_LABELS = {
    "land_surface_condition": {
        "n": "Flat",
        "o": "Moderate slope / obstructed",
        "t": "Steep slope / terraced",
    },
    "foundation_type": {
        "h": "Mud mortar - stone/brick",
        "i": "Bamboo/timber",
        "r": "Cement/RC",
        "u": "Unknown",
        "w": "Other",
    },
    "roof_type": {
        "n": "Bamboo/timber - light roof",
        "q": "Bamboo/timber - heavy roof",
        "x": "RCC/RB/RBC",
    },
    "ground_floor_type": {
        "f": "Mud",
        "m": "Timber",
        "v": "RCC",
        "x": "Brick/stone",
        "z": "Other",
    },
    "other_floor_type": {
        "j": "Timber/bamboo - mud",
        "q": "Timber plank",
        "s": "RCC/RB/RBC",
        "x": "Not applicable",
    },
    "position": {
        "j": "Attached on one side",
        "o": "Not attached / stand-alone",
        "s": "Attached on two sides",
        "t": "Attached on three sides / corner",
    },
    "plan_configuration": {
        "a": "Square",
        "c": "Rectangular",
        "d": "T-shaped / cross-shaped",
        "f": "L-shaped",
        "m": "Multi-projected",
        "n": "T-shaped",
        "o": "Other",
        "q": "U-shaped",
        "s": "Irregular",
        "u": "Unknown",
    },
    "legal_ownership_status": {
        "a": "Private",
        "r": "Rented",
        "v": "Other",
        "w": "Government",
    },
}

# Column -> friendly display name
FEATURE_DISPLAY_NAMES = {
    "building_id": "Building ID",
    "geo_level_1_id": "Geo Level 1",
    "geo_level_2_id": "Geo Level 2",
    "geo_level_3_id": "Geo Level 3",
    "count_floors_pre_eq": "Floor Count (pre-earthquake)",
    "age": "Building Age",
    "area_percentage": "Area Percentage",
    "height_percentage": "Height Percentage",
    "land_surface_condition": "Land Surface Condition",
    "foundation_type": "Foundation Type",
    "roof_type": "Roof Type",
    "ground_floor_type": "Ground Floor Type",
    "other_floor_type": "Other Floor Type",
    "position": "Building Position",
    "plan_configuration": "Plan Configuration",
    "legal_ownership_status": "Legal Ownership Status",
    "count_families": "Number of Families",
    "damage_grade": "Damage Grade",
}

CATEGORICAL_FEATURES = list(CATEGORICAL_LABELS.keys())

NUMERIC_FEATURES = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families",
]

SUPERSTRUCTURE_FLAGS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
]

SECONDARY_USE_FLAGS = [
    "has_secondary_use",
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other",
]

BINARY_FLAGS = SUPERSTRUCTURE_FLAGS + SECONDARY_USE_FLAGS

GEO_FEATURES = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

# Features offered in the single "Analyze a feature" selector on the
# Feature Analysis page. Geography has its own drill-down page (sunburst)
# because geo_level_2/3 have too many distinct values for a value-count view.
ANALYZABLE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FLAGS

_SUPERSTRUCTURE_MATERIAL = {
    "adobe_mud": "Adobe/mud",
    "mud_mortar_stone": "Mud mortar - stone",
    "stone_flag": "Stone flag",
    "cement_mortar_stone": "Cement mortar - stone",
    "mud_mortar_brick": "Mud mortar - brick",
    "cement_mortar_brick": "Cement mortar - brick",
    "timber": "Timber",
    "bamboo": "Bamboo",
    "rc_non_engineered": "RC, non-engineered",
    "rc_engineered": "RC, engineered",
    "other": "other material",
}

_SECONDARY_USE = {
    "agriculture": "agriculture",
    "hotel": "a hotel",
    "rental": "rental",
    "institution": "an institution",
    "school": "a school",
    "industry": "industry",
    "health_post": "a health post",
    "gov_office": "a government office",
    "use_police": "a police station",
    "other": "another secondary purpose",
}


def _flag_description(col: str) -> str:
    if col == "has_secondary_use":
        return "Whether the building has any secondary use beyond a dwelling."
    if col.startswith("has_superstructure_"):
        material = _SUPERSTRUCTURE_MATERIAL.get(col.replace("has_superstructure_", ""), "")
        return f"Whether the superstructure includes {material}."
    if col.startswith("has_secondary_use_"):
        use = _SECONDARY_USE.get(col.replace("has_secondary_use_", ""), "")
        return f"Whether the building is also used for {use}."
    return ""


FEATURE_DESCRIPTIONS = {
    "building_id": "Unique identifier for the building (primary key).",
    "geo_level_1_id": "Geographic region identifier, level 1 (largest area).",
    "geo_level_2_id": "Geographic region identifier, level 2 (nested within level 1).",
    "geo_level_3_id": "Geographic region identifier, level 3 (most granular, nested within level 2).",
    "count_floors_pre_eq": "Number of floors the building had before the earthquake.",
    "age": "Age of the building, in years.",
    "area_percentage": "Normalised footprint area of the building.",
    "height_percentage": "Normalised height of the building.",
    "land_surface_condition": "Surface condition of the land the building sits on.",
    "foundation_type": "Type of foundation used.",
    "roof_type": "Type of roof used.",
    "ground_floor_type": "Construction material of the ground floor.",
    "other_floor_type": "Construction material of the floor(s) above ground level.",
    "position": "Position of the building relative to adjacent structures.",
    "plan_configuration": "Shape of the building's floor plan.",
    "legal_ownership_status": "Legal ownership status of the land.",
    "count_families": "Number of families living in the building.",
    "damage_grade": "Target variable - earthquake damage grade (1 = low, 2 = medium, 3 = severe/destroyed).",
}
for _flag in BINARY_FLAGS:
    FEATURE_DESCRIPTIONS[_flag] = _flag_description(_flag)


def display_name(col: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title())


def code_to_label(col: str, code) -> str:
    return CATEGORICAL_LABELS.get(col, {}).get(code, str(code))


def feature_group(col: str) -> str:
    if col == "building_id":
        return "Identifier"
    if col == "damage_grade":
        return "Target"
    if col in GEO_FEATURES:
        return "Geography"
    if col in CATEGORICAL_FEATURES:
        return "Categorical"
    if col in NUMERIC_FEATURES:
        return "Numeric"
    if col in BINARY_FLAGS:
        return "Binary flag"
    return "Other"
