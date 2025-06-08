import ast
import kagglehub
import polars as pl
from files import Dota2Files, get_lf


def get_heroes(path: str):

    fixed_hero_cols = ["primary_attribute", "attack_type", "roles_vector"]
    hero_cols = ["base_health", "base_health_regen", "base_mana", "base_mana_regen",
                 "base_armor", "base_mr", "base_attack_min", "base_attack_max",
                 "base_str", "base_agi", "base_int",
                 "str_gain", "agi_gain", "int_gain",
                 "attack_range", "attack_rate",]

    heroes = get_lf(Dota2Files.HEROES, path)
    attributes = heroes.select(
        pl.col("primary_attr")).unique().collect().to_dict(as_series=False)["primary_attr"]
    dict_attributes = {attr: i for i, attr in enumerate(attributes)}

    roles: list[str] = list({role for roles_list in heroes.select("roles").collect().to_series().to_list() for role in (
        ast.literal_eval(roles_list) if isinstance(roles_list, str) else roles_list)})
    dict_roles = {role: i for i, role in enumerate(roles)}
    roles_idx = [i for i in dict_roles.values()]

    hero_ids = heroes.select(pl.col("id")).unique().sort("id").collect().to_series().to_list()
    dict_hero_index = {hid: i for i, hid in enumerate(hero_ids)}

    return (
        heroes
        .with_columns(
            pl.col("primary_attr").map_elements(lambda x: dict_attributes.get(x) if isinstance(
                x, str) else x, return_dtype=pl.UInt32).alias("primary_attribute"),
            pl.col("roles").map_elements(
                lambda x: [dict_roles.get(y) for y in ast.literal_eval(
                    x)] if isinstance(x, str) else x,
                return_dtype=pl.List(pl.UInt32)
            ),
            pl.col("attack_type").map_elements(
                lambda x: 0 if x == "Melee" else 1 if x == "Ranged" else None, return_dtype=pl.UInt8
            ).alias("attack_type"),
            
            
            *[pl.col(col).cast(pl.Float64, strict=False).fill_null(strategy="zero").alias(col) for col in hero_cols],
            pl.col("id").map_elements(lambda x: dict_hero_index.get(x), return_dtype=pl.UInt32).alias("hero_idx"),
        )
        .with_columns(
            pl.col("roles").map_elements(
                lambda x: [1 if i in x else 0 for i in roles_idx],
                return_dtype=pl.List(pl.UInt32)
            ).alias("roles_vector")
        )
        .select(
            pl.col("id").alias("hero_id").cast(pl.Int32),
            pl.col("localized_name").alias("hero_name"),
            pl.col("roles"),
            pl.col("hero_idx"),  # Seleciona a nova coluna
            *fixed_hero_cols,
            *hero_cols
        )

    ), hero_cols, dict_attributes, dict_roles

if __name__ == "__main__":
    dataset_name = "bwandowando/dota-2-pro-league-matches-2023"
    path = kagglehub.dataset_download(dataset_name)
    heroes, hero_cols, dict_attributes, dict_roles = get_heroes(path)
    print(heroes.select(pl.col("roles_vector").list.len()).collect())
    print(hero_cols)
    print(dict_attributes)
    print(dict_roles)