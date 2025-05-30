metadata_cols = ["version", "match_id", "leagueid", "start_date_time", "duration", "cluster", "replay_salt",
                 "radiant_win",
                 "pre_game_duration", "match_seq_num", "tower_status_radiant", "tower_status_dire",
                 "barracks_status_radiant", "barracks_status_dire", "first_blood_time", "lobby_type", "human_players",
                 "game_mode", "flags", "engine", "radiant_score", "dire_score", "radiant_team_id", "radiant_logo",
                 "radiant_team_complete", "dire_team_id", "dire_logo", "dire_team_complete", "radiant_captain",
                 "dire_captain", "patch", "region", "throw", "loss", "comeback", "stomp", "replay_url", "series_id",
                 "series_type"]

draft_cols = ["order", "pick", "active_team", "hero_id", "player_slot", "extra_time",
              "total_time_taken", "match_id", "leagueid"]

objectives_cols = ["time", "type", "value", "killer", "team", "slot",
                   "key", "player_slot", "unit", "match_id", "leagueid"]

picks_bans_cols = ["is_pick", "hero_id",
                   "team", "order", "match_id", "leagueid"]

players_cols = ["player_slot", "obs_placed", "sen_placed", "creeps_stacked", "camps_stacked", "rune_pickups",
                "firstblood_claimed", "teamfight_participation", "towers_killed", "roshans_killed", "observers_placed",
                "stuns", "max_hero_hit", "times", "gold_t", "lh_t", "dn_t", "xp_t", "obs_log", "sen_log",
                "obs_left_log", "sen_left_log", "purchase_log", "kills_log", "buyback_log", "runes_log",
                "connection_log", "lane_pos", "obs", "sen", "actions", "pings", "purchase", "gold_reasons",
                "xp_reasons", "killed", "item_uses", "ability_uses", "ability_targets", "damage_targets", "hero_hits",
                "damage", "damage_taken", "damage_inflictor", "runes", "killed_by", "kill_streaks", "multi_kills",
                "life_state", "healing", "damage_inflictor_received", "randomed", "pred_vict", "party_id",
                "permanent_buffs", "party_size", "account_id", "team_number", "team_slot", "hero_id", "item_0",
                "item_1", "item_2", "item_3", "item_4", "item_5", "backpack_0", "backpack_1", "backpack_2",
                "item_neutral", "kills", "deaths", "assists", "leaver_status", "last_hits", "denies", "gold_per_min",
                "xp_per_min", "level", "net_worth", "aghanims_scepter", "aghanims_shard", "moonshard", "hero_damage",
                "tower_damage", "hero_healing", "gold", "gold_spent", "ability_upgrades_arr", "personaname", "name",
                "last_login", "radiant_win", "start_time", "duration", "cluster", "lobby_type", "game_mode",
                "is_contributor", "patch", "region", "isRadiant", "win", "lose", "total_gold", "total_xp",
                "kills_per_min", "kda", "abandons", "neutral_kills", "tower_kills", "courier_kills", "lane_kills",
                "hero_kills", "observer_kills", "sentry_kills", "roshan_kills", "necronomicon_kills", "ancient_kills",
                "buyback_count", "observer_uses", "sentry_uses", "lane_efficiency", "lane_efficiency_pct", "lane",
                "lane_role", "is_roaming", "purchase_time", "first_purchase_time", "item_win", "item_usage",
                "purchase_tpscroll", "actions_per_min", "life_state_dead", "rank_tier", "is_subscriber", "cosmetics",
                "benchmarks", "purchase_ward_observer", "purchase_ward_sentry", "purchase_gem", "purchase_rapier",
                "match_id", "leagueid", "performance_others", "additional_units", "repicked", "hero_variant",
                "neutral_tokens_log"]

exp_adv_cols = ["minute", "exp", "match_id", "leagueid"]

gold_adv_cols = ["minute", "gold", "match_id", "leagueid"]

team_fights_cols = ["start", "end", "last_death",
                    "deaths", "players", "match_id", "leagueid"]

leagues_cols = ["leagueid", "leaguename", "tier"]
heroes_cols = ["", "id", "name", "primary_attr", "attack_type", "roles", "img", "icon",
               "base_health", "base_health_regen", "base_mana", "base_mana_regen",
               "base_armor", "base_mr", "base_attack_min", "base_attack_max",
               "base_str", "base_agi",
               "base_int", "str_gain", "agi_gain", "int_gain", "attack_range",
               "projectile_speed", "attack_rate", "base_attack_time", "attack_point",
               "move_speed", "turn_rate", "cm_enabled", "legs", "day_vision",
               "night_vision", "localized_name"]
