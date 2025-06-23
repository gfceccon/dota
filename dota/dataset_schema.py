import polars as pl

# Schema: Define os esquemas (tipos de dados) para os principais arquivos do dataset de Dota 2.
# Cada dicionário representa o nome da coluna e o tipo esperado para facilitar a leitura e validação dos dados.


class Dota2DatasetSchema:
    # Esquema dos metadados das partidas
    metadata_schema = {
        'version': pl.Int64,
        'match_id': pl.Int64,
        'leagueid': pl.Int64,
        'game_mode': pl.Int64,

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

    # Esquema dos objetivos das partidas (eventos importantes)
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

    # Mapeamento de times para objetivos
    objectives_teams = {
        '2.0': 0,
        '3.0': 1,
    }

    # Mapeamento de tipos de objetivos para nomes mais legíveis
    objectives_types = {
        'CHAT_MESSAGE_AEGIS': 'aegis',
        'CHAT_MESSAGE_AEGIS_STOLEN': 'aegis_stolen',
        'CHAT_MESSAGE_COURIER_LOST': 'courier_lost',
        'CHAT_MESSAGE_DENIED_AEGIS': 'denied_aegis',
        'CHAT_MESSAGE_FIRSTBLOOD': 'first_blood',
        'CHAT_MESSAGE_ROSHAN_KILL': 'roshan_kill',
        'building_kill': 'building_kill',
    }

    # Esquema de picks e bans (seleção e banimento de heróis)
    picks_bans_schema = {
        'match_id': pl.Int64,
        'is_pick': pl.Boolean,
        'hero_id': pl.Int64,
        'team': pl.Int64,
        # 'order': pl.Int64,
    }

    # Esquema dos jogadores e suas estatísticas
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

    # Mapeamento de times para jogadores
    players_teams = {
        '2.0': 0,
        '3.0': 1,
    }

    # Esquema de vantagem de experiência
    exp_adv_schema = {
        'minute': pl.Int64,
        'exp': pl.Float64,
        'match_id': pl.Int64,
    }

    # Esquema de vantagem de ouro
    gold_adv_schema = {
        'minute': pl.Int64,
        'gold': pl.Float64,
        'match_id': pl.Int64,
    }

    # Esquema das ligas
    leagues_schema = {
        'leagueid': pl.Int64,
        'leaguename': pl.String,
        'tier': pl.String,
    }

    # Esquema dos patches do jogo
    patches_schema = {
        'patch': pl.Float64,
        'date': pl.String,
    }

    # Esquema dos itens do jogo
    items_schema = {
        'id': pl.Int64,
        'dname': pl.String,

        'cost': pl.Int64,

        'manacost': pl.Int64,
        'cooldown': pl.Int64,
    }

    # Mapeamento de player_slot para time (Radiant/Dire)
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

    # Esquema dos heróis (atributos base)
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

    # Esquema dos heróis já processados
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

    # Esquema dos metadados já processados
    metadata_parsed_schema = {
        # Metadata
        #'version': pl.Int64,
        'match_id': pl.Int64,

        'start_date_time': pl.Datetime,
        'duration': pl.Int64,

        'patch': pl.Int64,
        # 'region': pl.String,
        # 'series_id': pl.String,
        # 'series_type': pl.String,

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

    # Esquema dos jogadores já processados
    players_parsed_schema = {
        'match_id': pl.Int64,
        # 'start_time': pl.Int64,
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
        #'is_pick': pl.Boolean,
        #'team': pl.Int64,
        # 'order': pl.Int64,
    }

    # Esquema dos bans já processados
    ban_parsed_schema = {
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

    # Lista final de colunas para o dataset consolidado
    final_schema = [
        'account_id_dire',
        'account_id_radiant',
        'ancient_kills_dire',
        'ancient_kills_radiant',
        'assists_dire',
        'assists_radiant',
        'attack_range_ban_dire',
        'attack_range_ban_radiant',
        'attack_range_dire',
        'attack_range_radiant',
        'attack_type_ban_dire',
        'attack_type_ban_radiant',
        'attack_type_dire',
        'attack_type_radiant',
        'backpack_vector_dire',
        'backpack_vector_radiant',
        'barracks_status_dire',
        'barracks_status_radiant',
        'buyback_count_dire',
        'buyback_count_radiant',
        'camps_stacked_dire',
        'camps_stacked_radiant',
        'courier_kills_dire',
        'courier_kills_radiant',
        'creeps_stacked_dire',
        'creeps_stacked_radiant',
        'day_vision_ban_dire',
        'day_vision_ban_radiant',
        'day_vision_dire',
        'day_vision_radiant',
        'deaths_dire',
        'deaths_radiant',
        'denies_dire',
        'denies_radiant',
        'dn_t_dire',
        'dn_t_radiant',
        'duration',
        'exp_adv',
        'firstblood_claimed_dire',
        'firstblood_claimed_radiant',
        'gold_adv',
        'gold_dire',
        'gold_per_min_dire',
        'gold_per_min_radiant',
        'gold_radiant',
        'gold_spent_dire',
        'gold_spent_radiant',
        'gold_t_dire',
        'gold_t_radiant',
        'hero_damage_dire',
        'hero_damage_radiant',
        'hero_healing_dire',
        'hero_healing_radiant',
        'hero_id_ban_dire',
        'hero_id_ban_radiant',
        'hero_id_dire',
        'hero_id_radiant',
        'hero_kills_dire',
        'hero_kills_radiant',
        'hero_name_ban_dire',
        'hero_name_ban_radiant',
        'hero_name_dire',
        'hero_name_radiant',
        'is_pick_dire',
        'is_pick_radiant',
        'item_neutral_dire',
        'item_neutral_radiant',
        'items_vector_dire',
        'items_vector_radiant',
        'kill_dire',
        'kills_radiant',
        'lane_kills_dire',
        'lane_kills_radiant',
        'last_hits_dire',
        'last_hits_radiant',
        'leagueid',
        'leaguename',
        'level_dire',
        'level_radiant',
        'lh_t_dire',
        'lh_t_radiant',
        'match_id',
        'match_id_dire',
        'match_id_radiant',
        'move_speed_ban_dire',
        'move_speed_ban_radiant',
        'move_speed_dire',
        'move_speed_radiant',
        'necronomicon_kills_dire',
        'necronomicon_kills_radiant',
        'net_worth_dire',
        'net_worth_radiant',
        'neutral_kills_dire',
        'neutral_kills_radiant',
        'night_vision_ban_dire',
        'night_vision_ban_radiant',
        'night_vision_dire',
        'night_vision_radiant',
        'obs_placed_dire',
        'obs_placed_radiant',
        'observer_kills_dire',
        'observer_kills_radiant',
        'patch',
        'player_slot_ban_dire',
        'player_slot_ban_radiant',
        'player_slot_dire',
        'player_slot_radiant',
        'primary_attribute_ban_dire',
        'primary_attribute_ban_radiant',
        'primary_attribute_dire',
        'primary_attribute_radiant',
        'purchase_gem_dire',
        'purchase_gem_radiant',
        'purchase_rapier_dire',
        'purchase_rapier_radiant',
        'purchase_ward_observer_dire',
        'purchase_ward_observer_radiant',
        'purchase_ward_sentry_dire',
        'purchase_ward_sentry_radiant',
        'radiant_win',
        'roles_vector_ban_dire',
        'roles_vector_ban_radiant',
        'roles_vector_dire',
        'roles_vector_radiant',
        'roshan_kills_dire',
        'roshan_kills_radiant',
        'roshans_killed_dire',
        'roshans_killed_radiant',
        'rune_pickups_dire',
        'rune_pickups_radiant',
        'sen_placed_dire',
        'sen_placed_radiant',
        'sentry_kills_dire',
        'sentry_kills_radiant',
        'start_date_time',
        'stuns_dire',
        'stuns_radiant',
        'team_dire',
        'team_radiant',
        'teamfight_participation_dire',
        'teamfight_participation_radiant',
        'tier',
        'times_dire',
        'times_radiant',
        'tower_damage_dire',
        'tower_damage_radiant',
        'tower_kills_dire',
        'tower_kills_radiant',
        'tower_status_dire',
        'tower_status_radiant',
        'towers_killed_dire',
        'towers_killed_radiant',
        'version',
        'xp_per_min_dire',
        'xp_per_min_radiant',
        'xp_t_dire',
        'xp_t_radiant'
    ]
