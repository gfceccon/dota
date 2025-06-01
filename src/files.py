year = 2024

metadata_file = f"{year}/main_metadata.csv"
draft_file = f"{year}/draft_timings.csv"
objectives_file = f"{year}/objectives.csv"
picks_bans_file = f"{year}/picks_bans.csv"
players_file = f"{year}/players.csv"
exp_adv_file = f"{year}/radiant_exp_adv.csv"
gold_adv_file = f"{year}/radiant_gold_adv.csv"
team_fights_file = f"{year}/teamfights.csv"
leagues_file = f"Constants/Constants.Leagues.csv"
heroes_file = f"Constants/Constants.Heroes.csv"
patches_file = f"Constants/Constants.Patch.csv"
patches_notes_file = f"Constants/Constants.PatchNotes.csv"

def get_metadata(year: str) -> str:
    return f"{year}/main_metadata.csv"