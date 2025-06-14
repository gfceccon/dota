metadata = [
    "version", "match_id", "leagueid", "start_date_time", "duration",
    "tower_status_radiant", "tower_status_dire",
    "radiant_win", "first_blood_time", "radiant_score",
    "barracks_status_radiant", "barracks_status_dire",  "dire_score",
    "radiant_team_id", "radiant_team_complete",
    "dire_team_id", "dire_team_complete",
    "radiant_captain", "dire_captain",
    "patch", "region", "series_id", "series_type"]

objectives = [
    "time", "type", "value", "killer", "team", "slot",
    "key", "player_slot", "unit", "match_id",]

obj_types = {
    "CHAT_MESSAGE_AEGIS": "aegis",
    "CHAT_MESSAGE_AEGIS_STOLEN": "aegis_stolen",
    "CHAT_MESSAGE_COURIER_LOST": "courier_lost",
    "CHAT_MESSAGE_DENIED_AEGIS": "denied_aegis",
    "CHAT_MESSAGE_FIRSTBLOOD": "first_blood",
    "CHAT_MESSAGE_ROSHAN_KILL": "roshan_kill",
    "building_kill": "building_kill",
}

picks_bans = ["is_pick", "hero_id", "team", "match_id", ]

players = [
    "obs_placed", "sen_placed",
    "creeps_stacked", "camps_stacked", "rune_pickups",
    "firstblood_claimed",
    "towers_killed", "roshans_killed", "observers_placed",
    "stuns", "max_hero_hit",
    "gold_t", "lh_t", "dn_t", "xp_t", "lane_pos",
    "obs", "sen",
    "damage", "damage_taken", "damage_inflictor", "runes",
    "healing", "damage_inflictor_received", "randomed", "pred_vict", "party_id",
    "permanent_buffs", "party_size", "account_id", "team_number", "team_slot", "hero_id", "item_0",
    "item_1", "item_2", "item_3", "item_4", "item_5", "backpack_0", "backpack_1", "backpack_2",
    "item_neutral",
    "kills", "deaths", "assists",
    "leaver_status", "last_hits", "denies",
    "gold_per_min", "xp_per_min", "level", "net_worth",
    "aghanims_scepter", "aghanims_shard", "moonshard", "hero_damage",
    "tower_damage", "hero_healing", "gold", "gold_spent", "total_gold", "total_xp",
    "kills_per_min", "neutral_kills", "tower_kills", "courier_kills",
    "hero_kills", "observer_kills", "sentry_kills", "roshan_kills",
    "necronomicon_kills", "ancient_kills", "buyback_count",
    "purchase_gem", "purchase_rapier", "match_id", ]

exp_adv = ["minute", "exp", "match_id",]

gold_adv = ["minute", "gold", "match_id",]

team_fights = ["start", "end", "last_death",
               "deaths", "players", "match_id",]

leagues = ["leagueid", "leaguename", "tier"]

heroes = [
    "id", "name", "primary_attr", "attack_type", "roles",
    "base_health", "base_health_regen", "base_mana", "base_mana_regen",
    "base_armor", "base_mr", "base_attack_min", "base_attack_max",
    "base_str", "base_agi",
    "base_int", "str_gain", "agi_gain", "int_gain", "attack_range",
    "move_speed", "day_vision", "night_vision",
    "localized_name"]
patches = ["patch","date"]