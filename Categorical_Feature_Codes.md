# Categorical Feature Codes

Expanded meanings and interpretations for the coded categorical values used in the earthquake damage dataset.

| Feature                    | Code | Expanded meaning                     | Interpretation                                            |
| --------------------------- | ---- | ------------------------------------- | ----------------------------------------------------------- |
| **land_surface_condition** | `n`  | Flat                                  | Building located on relatively flat terrain                 |
|                              | `o`  | Moderate slope / obstructed           | Sloping or locally obstructed terrain                       |
|                              | `t`  | Steep slope / terraced                | Steeper or terraced terrain                                 |
| **foundation_type**         | `h`  | Mud mortar–stone/brick                | Masonry foundation using mud mortar                         |
|                              | `i`  | Bamboo/timber                         | Timber/bamboo foundation                                    |
|                              | `r`  | Cement/RC                             | Reinforced-concrete foundation                               |
|                              | `u`  | Unknown                               | Foundation type not known/recorded                          |
|                              | `w`  | Other                                 | Other foundation type                                       |
| **roof_type**                | `n`  | Bamboo/timber – light roof            | Lightweight bamboo/timber roof                               |
|                              | `q`  | Bamboo/timber – heavy roof            | Heavier bamboo/timber roof                                   |
|                              | `x`  | RCC/RB/RBC                            | Reinforced concrete roof/slab                                |
| **ground_floor_type**       | `f`  | Mud                                   | Mud construction                                             |
|                              | `m`  | Timber                                | Timber construction                                          |
|                              | `v`  | RCC                                   | Reinforced-concrete construction                             |
|                              | `x`  | Brick/stone                           | Masonry construction                                          |
|                              | `z`  | Other                                 | Other construction                                            |
| **other_floor_type**        | `j`  | Timber/bamboo – mud                   | Timber/bamboo floors with mud construction                    |
|                              | `q`  | Timber planck                         | Timber plank floor                                             |
|                              | `s`  | RCC/RB/RBC                            | Reinforced concrete floor/slab                                 |
|                              | `x`  | Not applicable                        | No upper floor / not applicable                                |
| **position**                 | `j`  | Attached on one side                  | Building attached to a neighbouring building on one side       |
|                              | `o`  | Not attached / stand-alone            | Building not attached to neighbouring buildings                |
|                              | `s`  | Attached on two sides                 | Buildings attached on two sides                                 |
|                              | `t`  | Attached on three sides / corner      | Building attached on three sides                                |
| **plan_configuration**       | `a`  | Square                                | Approximately square building plan                             |
|                              | `c`  | Rectangular                           | Rectangular plan                                                |
|                              | `d`  | T-shaped / cross-shaped               | More complex cross-type configuration                            |
|                              | `f`  | L-shaped                              | L-shaped plan                                                   |
|                              | `m`  | Multi-projected                       | Irregular/multi-projection configuration                        |
|                              | `n`  | T-shaped                              | T-type configuration                                             |
|                              | `o`  | Other                                 | Other plan configuration                                        |
|                              | `q`  | U-shaped                              | U-type configuration                                             |
|                              | `s`  | Irregular                             | Irregular configuration                                          |
|                              | `u`  | Unknown                               | Plan configuration unknown                                       |
| **legal_ownership_status**  | `a`  | Private                               | Privately owned land                                            |
|                              | `r`  | Rented                                | Rented/leased land                                               |
|                              | `v`  | Other                                 | Other ownership arrangement                                      |
|                              | `w`  | Government                            | Government/public land                                          |
