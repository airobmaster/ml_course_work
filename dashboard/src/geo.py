"""Build the sunburst node table (id / label / parent / buildings /
avg_damage_grade / hover) for the full Geo Level 1 -> 2 -> 3 hierarchy,
mirroring the node-building logic and hover-breadcrumb style used in the
source notebook's EDA."""

import pandas as pd

from src import queries as q


def full_hierarchy_nodes() -> pd.DataFrame:
    geo1 = q.geo1_summary()
    geo2 = q.geo2_summary_all()
    geo3 = q.geo3_summary_all()

    rows = []

    for _, r in geo1.iterrows():
        rows.append(
            {
                "id": f"geo1_{r.geo_level_1_id}",
                "label": f"geo_level_1_id = {r.geo_level_1_id}",
                "parent": "",
                "buildings": r.buildings,
                "avg_damage_grade": r.avg_damage_grade,
                "hover": f"<b>geo_level_1_id = {r.geo_level_1_id}</b>",
            }
        )

    for _, r in geo2.iterrows():
        rows.append(
            {
                "id": f"geo2_{r.geo_level_1_id}_{r.geo_level_2_id}",
                "label": f"geo_level_2_id = {r.geo_level_2_id}",
                "parent": f"geo1_{r.geo_level_1_id}",
                "buildings": r.buildings,
                "avg_damage_grade": r.avg_damage_grade,
                "hover": (
                    f"<b>geo_level_2_id = {r.geo_level_2_id}</b>"
                    f"<br>geo_level_1_id = {r.geo_level_1_id}"
                ),
            }
        )

    for _, r in geo3.iterrows():
        rows.append(
            {
                "id": f"geo3_{r.geo_level_1_id}_{r.geo_level_2_id}_{r.geo_level_3_id}",
                "label": f"geo_level_3_id = {r.geo_level_3_id}",
                "parent": f"geo2_{r.geo_level_1_id}_{r.geo_level_2_id}",
                "buildings": r.buildings,
                "avg_damage_grade": r.avg_damage_grade,
                "hover": (
                    f"<b>geo_level_3_id = {r.geo_level_3_id}</b>"
                    f"<br>geo_level_2_id = {r.geo_level_2_id}"
                    f"<br>geo_level_1_id = {r.geo_level_1_id}"
                ),
            }
        )

    return pd.DataFrame(rows)
