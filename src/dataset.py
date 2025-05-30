import kagglehub
import polars as pl
from matches import get_matches
from players import get_players_draft
from objectives import get_objectives


dataset_name = "bwandowando/dota-2-pro-league-matches-2023/versions/177"
path = kagglehub.dataset_download(dataset_name)

matches = get_matches(path, [54], ["professional"])
players_draft = get_players_draft(path=path)
objectives = get_objectives(path=path)