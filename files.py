from enum import Enum
import polars as pl

# Enum para os arquivos
class Dota2Files(Enum):
    METADATA = f"/main_metadata.csv"
    DRAFT = f"/draft_timings.csv"
    OBJECTIVES = f"/objectives.csv"
    PICKS_BANS = f"/picks_bans.csv"
    PLAYERS = f"/players.csv"
    EXP_ADV = f"/radiant_exp_adv.csv"
    GOLD_ADV = f"/radiant_gold_adv.csv"
    TEAM_FIGHTS = f"/teamfights.csv"
    LEAGUES = f"Constants/Constants.Leagues.csv"
    HEROES = f"Constants/Constants.Heroes.csv"
    PATCHES = f"Constants/Constants.Patch.csv"
    PATCHES_NOTES = f"Constants/Constants.PatchNotes.csv"

# Retorna o LazyFrame para o arquivo especificado
def get_lf(file:Dota2Files, path: str , years: tuple[int, int] = (2020, 2024)) -> pl.LazyFrame:
    if(file == Dota2Files.LEAGUES or
       file == Dota2Files.HEROES or
       file == Dota2Files.PATCHES or
       file == Dota2Files.PATCHES_NOTES):
        return pl.scan_csv(f"{path}/{file.value}")
    scans: list[pl.LazyFrame]= []
    schemas: list[pl.Schema] = []
    for year in range(years[0], years[1]):
        df = pl.scan_csv(f"{path}/{year}/{file.value}")
        scans.append(df)
        schemas.append(df.collect_schema())
    names = set(schemas[0].names())
    for schema in schemas:
        names.intersection_update(schema.names())
    lf = pl.concat(scans, how="diagonal_relaxed").lazy()
    return lf
    