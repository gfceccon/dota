
from files import heroes_file
import ast
import polars as pl
def get_heroes(path: str):
    heroes = pl.read_csv(f"{path}/{heroes_file}")
    attributes = heroes.select(pl.col("primary_attr")).unique().to_series().to_list()
    dict_attributes = {attr: i for i, attr in enumerate(attributes)}
    
    roles = list({role for roles_list in heroes['roles'] for role in (
        ast.literal_eval(roles_list) if isinstance(roles_list, str) else roles_list)})
    dict_roles = {role: i for i, role in enumerate(roles)}
    
    return heroes.with_columns(
        pl.col("primary_attr").map_elements(lambda x: dict_attributes[x] if isinstance(x, str) else x).alias("primary_attr"),
        pl.col("roles").map_elements(lambda x: [dict_roles[x] for x in ast.literal_eval(x)] if isinstance(x, str) else x).alias("roles")
    ).select(
        pl.col("id").alias("hero_id"),
        pl.col("name").alias("hero_name"),
        pl.col("primary_attr").alias("hero_primary_attr"),
        pl.col("attack_type").alias("hero_attack_type"),
        pl.col("roles").alias("hero_roles"),
    )