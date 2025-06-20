import polars as pl


class Schema:

    metadata_schema = {
        'version': pl.Int64,
        'match_id': pl.Int64,
        'leagueid': pl.Int64,

        'start_date_time': pl.Date,
        'duration': pl.Int64,

        'patch': pl.Int64,
        'region': pl.String,
        'series_id': pl.String,
        'series_type': pl.String,

        'radiant_win': pl.Boolean,

        'tower_status_radiant': pl.String,
        'tower_status_dire': pl.String,

        'barracks_status_radiant': pl.String,
        'barracks_status_dire': pl.String,


        'radiant_team_complete': pl.Boolean,
        'dire_team_complete': pl.Boolean,


        'radiant_team_id': pl.Int64,
        'radiant_captain': pl.Float64,

        'dire_team_id': pl.Float64,
        'dire_captain': pl.Float64,
    }

    objectives_schema = {
        'time': pl.Int64,
        'type': pl.String,
        'killer': pl.Int64,
        'team': pl.String,

        'value': pl.Int64,
        'key': pl.String,

        'player_slot': pl.String,
        'slot': pl.String,

        'unit': pl.String,

        'match_id': pl.Int64,
        'leagueid': pl.Int64,
    }

    objectives_teams = {
        '2.0': 0,
        '3.0': 1,
    }

    objectives_types = {
        'CHAT_MESSAGE_AEGIS': 'aegis',
        'CHAT_MESSAGE_AEGIS_STOLEN': 'aegis_stolen',
        'CHAT_MESSAGE_COURIER_LOST': 'courier_lost',
        'CHAT_MESSAGE_DENIED_AEGIS': 'denied_aegis',
        'CHAT_MESSAGE_FIRSTBLOOD': 'first_blood',
        'CHAT_MESSAGE_ROSHAN_KILL': 'roshan_kill',
        'building_kill': 'building_kill',
    }

    picks_bans_schema = {
        'match_id': pl.Int64,
        'is_pick': pl.Boolean,
        'hero_id': pl.Int64,
        'team': pl.Int64,
        'order': pl.Int64,
    }

    players_schema = {
        'match_id': pl.Int64,
        'start_time': pl.Int64,
        'duration': pl.Int64,
        'player_slot': pl.Int64,
        'obs_placed': pl.Int64,
        'sen_placed': pl.Int64,
        'creeps_stacked': pl.Int64,
        'camps_stacked': pl.Int64,
        'rune_pickups': pl.Int64,
        'firstblood_claimed': pl.Boolean,
        'teamfight_participation': pl.Float64,
        'towers_killed': pl.Int64,
        'roshans_killed': pl.Int64,
        'observers_placed': pl.Int64,
        'stuns': pl.Float64,
        'times': pl.List(pl.Float64),
        'gold_t': pl.List(pl.Float64),
        'lh_t': pl.List(pl.Float64),
        'dn_t': pl.List(pl.Float64),
        'xp_t': pl.List(pl.Float64),
        'randomed': pl.Boolean,
        'pred_vict': pl.Boolean,
        'party_id': pl.Int64,
        'permanent_buffs': pl.String,
        'party_size': pl.Int64,
        'account_id': pl.Int64,
        'team_number': pl.Int64,
        'team_slot': pl.Int64,
        'hero_id': pl.Int64,
        'item_0': pl.Int64,
        'item_1': pl.Int64,
        'item_2': pl.Int64,
        'item_3': pl.Int64,
        'item_4': pl.Int64,
        'item_5': pl.Int64,
        'backpack_0': pl.Int64,
        'backpack_1': pl.Int64,
        'backpack_2': pl.Int64,
        'item_neutral': pl.Int64,
        'kills': pl.Int64,
        'deaths': pl.Int64,
        'assists': pl.Int64,
        'last_hits': pl.Int64,
        'denies': pl.Int64,
        'gold_per_min': pl.Int64,
        'xp_per_min': pl.Int64,
        'level': pl.Int64,
        'net_worth': pl.Int64,
        'hero_damage': pl.Int64,
        'tower_damage': pl.Int64,
        'hero_healing': pl.Int64,
        'gold': pl.Int64,
        'gold_spent': pl.Int64,
        'neutral_kills': pl.Int64,
        'tower_kills': pl.Int64,
        'courier_kills': pl.Int64,
        'lane_kills': pl.Int64,
        'hero_kills': pl.Int64,
        'observer_kills': pl.Int64,
        'sentry_kills': pl.Int64,
        'roshan_kills': pl.Int64,
        'necronomicon_kills': pl.Int64,
        'ancient_kills': pl.Int64,
        'buyback_count': pl.Int64,
        'observer_uses': pl.Int64,
        'sentry_uses': pl.Int64,
        'purchase_ward_observer': pl.Float64,
        'purchase_ward_sentry': pl.Float64,
        'purchase_gem': pl.Float64,
        'purchase_rapier': pl.Float64,
    }

    # Radiant e Dire
    players_teams = {
        '2.0': 0,
        '3.0': 1,
    }

    exp_adv_schema = {
        'minute': pl.Int64,
        'exp': pl.Float64,
        'match_id': pl.Int64,
    }

    gold_adv_schema = {
        'minute': pl.Int64,
        'gold': pl.Float64,
        'match_id': pl.Int64,
    }

    leagues_schema = {
        'leagueid': pl.Int64,
        'leaguename': pl.String,
        'tier': pl.String,
    }

    patches_schema = {
        'patch': pl.Float64,
        'date': pl.String,
    }

    items_schema = {
        'id': pl.Int64,
        'dname': pl.String,

        'cost': pl.Int64,

        'manacost': pl.Int64,
        'cooldown': pl.Int64,
    }

    player_slot_team = {
        # Radiant
        0.0: 0,
        1.0: 0,
        2.0: 0,
        3.0: 0,
        4.0: 0,
        # Dire
        128.0: 1,
        129.0: 1,
        130.0: 1,
        131.0: 1,
        132.0: 1,
    }

    heroes_schema = {
        'base_health': pl.Float64,
        'base_health_regen': pl.Float64,
        'base_mana': pl.Float64,
        'base_mana_regen': pl.Float64,
        'base_armor': pl.Float64,
        'base_mr': pl.Float64,
        'base_attack_min': pl.Float64,
        'base_attack_max': pl.Float64,
        'base_str': pl.Float64,
        'base_agi': pl.Float64,
        'base_int': pl.Float64,
        'str_gain': pl.Float64,
        'agi_gain': pl.Float64,
        'int_gain': pl.Float64,
        'attack_range': pl.Float64,
        'move_speed': pl.Float64,
        'day_vision': pl.Float64,
        'night_vision': pl.Float64,
    }

    heroes_parsed_schema = {
        'hero_id': pl.Float64,
        'hero_name': pl.String,
        'primary_attribute': pl.Int64,
        'attack_type': pl.Int64,
        'roles_vector': pl.List(pl.Int32),
        # 'base_health': pl.Float64,
        # 'base_health_regen': pl.Float64,
        # 'base_mana': pl.Float64,
        # 'base_mana_regen': pl.Float64,
        # 'base_armor': pl.Float64,
        # 'base_mr': pl.Float64,
        # 'base_attack_min': pl.Float64,
        # 'base_attack_max': pl.Float64,
        # 'base_str': pl.Float64,
        # 'base_agi': pl.Float64,
        # 'base_int': pl.Float64,
        # 'str_gain': pl.Float64,
        # 'agi_gain': pl.Float64,
        # 'int_gain': pl.Float64,
        'attack_range': pl.Float64,
        'move_speed': pl.Float64,
        'day_vision': pl.Float64,
        'night_vision': pl.Float64,
    }

    metadata_parsed_schema = {
        # Metadata
        'version': pl.Int64,
        'match_id': pl.Int64,
        'leagueid': pl.Int64,

        'start_date_time': pl.Datetime,
        'duration': pl.Int64,

        'patch': pl.Int64,
        'region': pl.String,
        'series_id': pl.String,
        'series_type': pl.String,

        'radiant_win': pl.Boolean,

        'tower_status_radiant': pl.String,
        'tower_status_dire': pl.String,

        'barracks_status_radiant': pl.String,
        'barracks_status_dire': pl.String,

        # League
        'leagueid': pl.Int64,
        'leaguename': pl.String,
        'tier': pl.String,

        # Gold and Experience Advantage
        'gold_adv': pl.List(pl.Float64),
        'exp_adv': pl.List(pl.Float64),
    }

    players_parsed_schema = {
        'match_id': pl.Int64,
        'start_time': pl.Int64,
        'player_slot': pl.Int64,
        'obs_placed': pl.Int64,
        'sen_placed': pl.Int64,
        'creeps_stacked': pl.Int64,
        'camps_stacked': pl.Int64,
        'rune_pickups': pl.Int64,
        'firstblood_claimed': pl.Boolean,
        'teamfight_participation': pl.Float64,
        'towers_killed': pl.Int64,
        'roshans_killed': pl.Int64,
        # 'observers_placed': pl.Int64,
        'stuns': pl.Float64,
        'times': pl.List(pl.Float64),
        'gold_t': pl.List(pl.Float64),
        'lh_t': pl.List(pl.Float64),
        'dn_t': pl.List(pl.Float64),
        'xp_t': pl.List(pl.Float64),
        # 'randomed': pl.Boolean,
        # 'pred_vict': pl.Boolean,
        # 'party_id': pl.Int64,
        # 'permanent_buffs': pl.String,
        # 'party_size': pl.Int64,
        'account_id': pl.Int64,
        # 'team_number': pl.Int64,
        # 'team_slot': pl.Int64,
        'hero_id': pl.Int64,
        'items_vector': pl.Int64,
        'backpack_vector': pl.Int64,
        'item_neutral': pl.Int64,
        'kills': pl.Int64,
        'deaths': pl.Int64,
        'assists': pl.Int64,
        'last_hits': pl.Int64,
        'denies': pl.Int64,
        'gold_per_min': pl.Int64,
        'xp_per_min': pl.Int64,
        'level': pl.Int64,
        'net_worth': pl.Int64,
        'hero_damage': pl.Int64,
        'tower_damage': pl.Int64,
        'hero_healing': pl.Int64,
        'gold': pl.Int64,
        'gold_spent': pl.Int64,
        'neutral_kills': pl.Int64,
        'tower_kills': pl.Int64,
        'courier_kills': pl.Int64,
        'lane_kills': pl.Int64,
        'hero_kills': pl.Int64,
        'observer_kills': pl.Int64,
        'sentry_kills': pl.Int64,
        'roshan_kills': pl.Int64,
        'necronomicon_kills': pl.Int64,
        'ancient_kills': pl.Int64,
        'buyback_count': pl.Int64,
        # 'observer_uses': pl.Int64,
        # 'sentry_uses': pl.Int64,
        'purchase_ward_observer': pl.Float64,
        'purchase_ward_sentry': pl.Float64,
        'purchase_gem': pl.Float64,
        'purchase_rapier': pl.Float64,

        # Picks and Bans
        'is_pick': pl.Boolean,
        'team': pl.Int64,
        'order': pl.Int64,
    }
