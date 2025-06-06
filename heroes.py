
from files import heroes_file
import ast
import polars as pl


def get_heroes(path: str):
    heroes = pl.scan_csv(f"{path}/{heroes_file}")
    attributes = heroes.select(
        pl.col("primary_attr")).unique().collect().to_dict(as_series=False)["primary_attr"]
    dict_attributes = {attr: i for i, attr in enumerate(attributes)}

    roles: list[str] = list({role for roles_list in heroes.select("roles").collect().to_series().to_list() for role in (
        ast.literal_eval(roles_list) if isinstance(roles_list, str) else roles_list)})
    dict_roles = {role: i for i, role in enumerate(roles)}

    return heroes.with_columns(
        pl.col("primary_attr").map_elements(lambda x: dict_attributes.get(x) if isinstance(
            x, str) else x, return_dtype=pl.UInt32).alias("primary_attribute"),
        pl.col("roles").map_elements(
            lambda x: [dict_roles.get(y) for y in ast.literal_eval(x)] if isinstance(x, str) else x,
            return_dtype=pl.List(pl.UInt32)
        ).alias("roles"),
        pl.col("attack_type").map_elements(
            lambda x: 0 if x == "Melee" else 1 if x == "Ranged" else None, return_dtype=pl.UInt8
        ).alias("attack_type"),
    ).select(
        pl.col("id").alias("hero_id"),
        pl.col("localized_name").alias("hero_name"),
        pl.col("primary_attribute"),
        pl.col("attack_type"),
        pl.col("roles"),
    ), dict_attributes, dict_roles
