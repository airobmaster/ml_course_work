# Feature Description

## Identifiers

| Feature | Type | Description | Possible Values |
|---|---|---|---|
| `geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id` | int | Geographic region in which the building exists, from largest (level 1) to most specific sub-region (level 3). | Level 1: 0–30<br>Level 2: 0–1427<br>Level 3: 0–12567 |

## Building Characteristics

| Feature | Type | Description | Possible Values |
|---|---|---|---|
| `count_floors_pre_eq` | int | Number of floors in the building before the earthquake. | — |
| `age` | int | Age of the building in years. | — |
| `area_percentage` | int | Normalized area of the building footprint. | — |
| `height_percentage` | int | Normalized height of the building footprint. | — |
| `land_surface_condition` | categorical | Surface condition of the land where the building was built. | `n`, `o`, `t` |
| `foundation_type` | categorical | Type of foundation used while building. | `h`, `i`, `r`, `u`, `w` |
| `roof_type` | categorical | Type of roof used while building. | `n`, `q`, `x` |
| `ground_floor_type` | categorical | Type of the ground floor. | `f`, `m`, `v`, `x`, `z` |
| `other_floor_type` | categorical | Type of construction used in floors higher than the ground floor (excluding roof). | `j`, `q`, `s`, `x` |
| `position` | categorical | Position of the building. | `j`, `o`, `s`, `t` |
| `plan_configuration` | categorical | Building plan configuration. | `a`, `c`, `d`, `f`, `m`, `n`, `o`, `q`, `s`, `u` |
| `legal_ownership_status` | categorical | Legal ownership status of the land where the building was built. | `a`, `r`, `v`, `w` |
| `count_families` | int | Number of families that live in the building. | — |

## Superstructure Material (binary flags)

Flag variables indicating the material(s) the superstructure was made of.

| Feature | Type | Description | Possible Values |
|---|---|---|---|
| `has_superstructure_adobe_mud` | binary | Adobe/Mud | `0`, `1` |
| `has_superstructure_mud_mortar_stone` | binary | Mud Mortar – Stone | `0`, `1` |
| `has_superstructure_stone_flag` | binary | Stone | `0`, `1` |
| `has_superstructure_cement_mortar_stone` | binary | Cement Mortar – Stone | `0`, `1` |
| `has_superstructure_mud_mortar_brick` | binary | Mud Mortar – Brick | `0`, `1` |
| `has_superstructure_cement_mortar_brick` | binary | Cement Mortar – Brick | `0`, `1` |
| `has_superstructure_timber` | binary | Timber | `0`, `1` |
| `has_superstructure_bamboo` | binary | Bamboo | `0`, `1` |
| `has_superstructure_rc_non_engineered` | binary | Non-engineered reinforced concrete | `0`, `1` |
| `has_superstructure_rc_engineered` | binary | Engineered reinforced concrete | `0`, `1` |
| `has_superstructure_other` | binary | Any other material | `0`, `1` |

## Secondary Use (binary flags)

Flag variables indicating whether the building was used for any secondary purpose.

| Feature | Description |
|---|---|
| `has_secondary_use` | Building was used for any secondary purpose. |
| `has_secondary_use_agriculture` | Used for agricultural purposes. |
| `has_secondary_use_hotel` | Used as a hotel. |
| `has_secondary_use_rental` | Used for rental purposes. |
| `has_secondary_use_institution` | Used as the location of any institution. |
| `has_secondary_use_school` | Used as a school. |
| `has_secondary_use_industry` | Used for industrial purposes. |
| `has_secondary_use_health_post` | Used as a health post. |
| `has_secondary_use_gov_office` | Used as a government office. |
| `has_secondary_use_use_police` | Used as a police station. |
| `has_secondary_use_other` | Used secondarily for other purposes. |
