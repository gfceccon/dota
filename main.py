from dota import Dota2
from dota import Dataset
import polars as pl

if __name__ == "__main__":
    dataset = Dataset()
    for year in range(2020, 2025):
        lf = dataset.get_year(year)
        df = lf.collect()
        print(f"Year {year} matches:", df.shape[0])
        df.write_json(f"dota2_matches_{year}.json")
        
        df = dataset._objectives(year).collect()
        df.write_json(f"dota2_{year}_objectives.json")
        df = dataset._exp_adv(year).collect()
        df.write_json(f"dota2_{year}_exp_adv.json")
        df = dataset._gold_adv(year).collect()
        df.write_json(f"dota2_{year}_gold_adv.json")
        df = dataset._team_fights(year).collect()
        df.write_json(f"dota2_{year}_team_fights.json")
        df = dataset._picks_bans(year).collect()
        df.write_json(f"dota2_{year}_picks_bans.json")
    
    df = dataset.heroes.collect()
    df.write_json("dota2_heroes.json")
    df = dataset.metadata.collect()
    df.write_json("dota2_metadata.json")
    df = dataset.leagues.collect()
    df.write_json("dota2_leagues.json")
    df = dataset.patches.collect()
    df.write_json("dota2_patches.json")