columns: dict[str, dict[str, bool]] = {
    "match_id": {
        'include': False,
        'normalize': False,
    },
    'player_slot': {
        'include': False,
        'normalize': False,
    },
    "start_time": {
        'include': False,
        'normalize': False,
    },
    "obs_placed": {
        'include': True,
        'normalize': True,
    },
    "sen_placed": {
        'include': True,
        'normalize': True,
    },
    "creeps_stacked": {
        'include': True,
        'normalize': True,
    },
    "camps_stacked": {
        'include': True,
        'normalize': True,
    },
    "rune_pickups": {
        'include': True,
        'normalize': True,
    },
    "firstblood_claimed": {
        'include': False,
        'normalize': False,
    },
    "teamfight_participation": {
        'include': True,
        'normalize': False,
    },
    "towers_killed": {
        'include': True,
        'normalize': True,
    },
    "roshans_killed": {
        'include': True,
        'normalize': True,
    },
    "stuns": {
        'include': True,
        'normalize': True,
    },
    "times": {
        'include': False,
        'normalize': False,
    },
    "gold_t": {
        'include': False,
        'normalize': False,
    },
    "lh_t": {
        'include': False,
        'normalize': False,
    },
    "dn_t": {
        'include': False,
        'normalize': False,
    },
    "xp_t": {
        'include': False,
        'normalize': False,
    },
    "party_id": {
        'include': False,
        'normalize': False,
    },
    "account_id": {
        'include': False,
        'normalize': False,
    },
    "hero_id": {
        'include': True,
        'normalize': False,
    },
    "items_vector": {
        'include': False,
        'normalize': False,
    },
    "backpack_vector": {
        'include': False,
        'normalize': False,
    },
    "item_neutral": {
        'include': False,
        'normalize': False,
    },
    "kills": {
        'include': True,
        'normalize': True,
    },
    "deaths": {
        'include': True,
        'normalize': True,
    },
    "assists": {
        'include': True,
        'normalize': True,
    },
    "last_hits": {
        'include': True,
        'normalize': True,
    },
    "denies": {
        'include': True,
        'normalize': True,
    },
    "gold_per_min": {
        'include': True,
        'normalize': True,
    },
    "xp_per_min": {
        'include': True,
        'normalize': True,
    },
    "level": {
        'include': True,
        'normalize': True,
    },
    "net_worth": {
        'include': True,
        'normalize': True,
    },
    "hero_damage": {
        'include': True,
        'normalize': True,
    },
    "tower_damage": {
        'include': True,
        'normalize': True,
    },
    "hero_healing": {
        'include': True,
        'normalize': True,
    },
    "gold": {
        'include': True,
        'normalize': True,
    },
    "gold_spent": {
        'include': True,
        'normalize': True,
    },
    "neutral_kills": {
        'include': True,
        'normalize': True,
    },
    "tower_kills": {
        'include': True,
        'normalize': True,
    },
    "courier_kills": {
        'include': True,
        'normalize': True,
    },
    "lane_kills": {
        'include': True,
        'normalize': True,
    },
    "hero_kills": {
        'include': True,
        'normalize': True,
    },
    "observer_kills": {
        'include': True,
        'normalize': True,
    },
    "sentry_kills": {
        'include': True,
        'normalize': True,
    },
    "roshan_kills": {
        'include': True,
        'normalize': True,
    },
    "necronomicon_kills": {
        'include': True,
        'normalize': True,
    },
    "ancient_kills": {
        'include': True,
        'normalize': True,
    },
    "buyback_count": {
        'include': True,
        'normalize': True,
    },
    "purchase_ward_observer": {
        'include': True,
        'normalize': True,
    },
    "purchase_ward_sentry": {
        'include': True,
        'normalize': True,
    },
    "purchase_gem": {
        'include': True,
        'normalize': True,
    },
    "purchase_rapier": {
        'include': True,
        'normalize': True,
    },
    "is_pick": {
        'include': True,
        'normalize': False,
    },
    "team": {
        'include': True,
        'normalize': False,
    },
    "order": {
        'include': True,
        'normalize': False,
    },
    "version": {
        'include': False,
        'normalize': True,
    },
    "leagueid": {
        'include': False,
        'normalize': True,
    },
    "start_date_time": {
        'include': False,
        'normalize': True,
    },
    "duration": {
        'include': False,
        'normalize': True,
    },
    "patch": {
        'include': False,
        'normalize': True,
    },
    "region": {
        'include': False,
        'normalize': True,
    },
    "series_id": {
        'include': False,
        'normalize': True,
    },
    "series_type": {
        'include': False,
        'normalize': True,
    },
    "radiant_win": {
        'include': False,
        'normalize': True,
    },
    "tower_status_radiant": {
        'include': False,
        'normalize': True,
    },
    "tower_status_dire": {
        'include': False,
        'normalize': True,
    },
    "barracks_status_radiant": {
        'include': False,
        'normalize': True,
    },
    "barracks_status_dire": {
        'include': False,
        'normalize': True,
    },
    "leaguename": {
        'include': False,
        'normalize': True,
    },
    "tier": {
        'include': False,
        'normalize': True,
    },
    "gold_adv": {
        'include': False,
        'normalize': True,
    },
    "exp_adv": {
        'include': False,
        'normalize': True,
    },
    "hero_name": {
        'include': False,
        'normalize': True,
    },
    "attack_type": {
        'include': False,
        'normalize': True,
    },
    "roles_vector": {
        'include': False,
        'normalize': True,
    },
    "attack_range": {
        'include': False,
        'normalize': True,
    },
    "move_speed": {
        'include': False,
        'normalize': True,
    },
    "day_vision": {
        'include': False,
        'normalize': True,
    },
    "night_vision": {
        'include': False,
        'normalize': True,
    },
}


columns_emb: dict[str, bool] = {
    "picks": True,
    "bans": True,
    "roles": True,
    "attributes": True,
    "attack": True,
    "items": True,
    "backpack": True,
    "neutral": True,
}


def embeddings(emb_config: dict[str, list[int]], emb_dim_config: dict[str, int]) -> dict[str, tuple[bool, list[int], int]]:
    return {
        "radiant_picks": (
            columns_emb['picks'], emb_config['emb_hero'], emb_dim_config['emb_pick_dim']),
        "dire_picks": (
            columns_emb['picks'], emb_config['emb_hero'], emb_dim_config['emb_pick_dim']),
        "radiant_bans": (
            columns_emb['bans'], emb_config['emb_hero'], emb_dim_config['emb_ban_dim']),
        "dire_bans": (
            columns_emb['bans'], emb_config['emb_hero'], emb_dim_config['emb_ban_dim']),
        "radiant_roles_picks": (
            columns_emb['roles'], emb_config['emb_role'], emb_dim_config['emb_role_dim']),
        "dire_roles_picks": (
            columns_emb['roles'], emb_config['emb_role'], emb_dim_config['emb_role_dim']),
        "radiant_roles_bans": (
            columns_emb['roles'], emb_config['emb_role'], emb_dim_config['emb_role_dim']),
        "dire_roles_bans": (
            columns_emb['roles'], emb_config['emb_role'], emb_dim_config['emb_role_dim']),
        "radiant_attributes": (
            columns_emb['attributes'], emb_config['emb_attr'], emb_dim_config['emb_attr_dim']),
        "dire_attributes": (
            columns_emb['attributes'], emb_config['emb_attr'], emb_dim_config['emb_attr_dim']),
        "radiant_attack": (
            columns_emb['attack'], emb_config['emb_attr'], emb_dim_config['emb_attr_dim']),
        "dire_attack": (
            columns_emb['attack'], emb_config['emb_attr'], emb_dim_config['emb_attr_dim']),
        "radiant_items": (
            columns_emb['items'], emb_config['emb_item'], emb_dim_config['emb_item_dim']),
        "dire_items": (
            columns_emb['items'], emb_config['emb_item'], emb_dim_config['emb_item_dim']),
        "radiant_backpack": (
            columns_emb['backpack'], emb_config['emb_item'], emb_dim_config['emb_item_dim']),
        "dire_backpack": (
            columns_emb['backpack'], emb_config['emb_item'], emb_dim_config['emb_item_dim']),
        "radiant_neutral": (
            columns_emb['neutral'], emb_config['emb_item'], emb_dim_config['emb_neutral_dim']),
        "dire_neutral": (
            columns_emb['neutral'], emb_config['emb_item'], emb_dim_config['emb_neutral_dim']),
    }

def calc_input_dim(n_players, n_bans, emb_size):
    dim = (
        2 * (n_players + n_bans) *
        sum([1 if columns[key] else 0 for key in columns.keys()]))
    if columns_emb['picks']:
        dim += emb_size['pick'] * 2 * n_players
    if columns_emb['bans']:
        dim += emb_size['bans'] * 2 * n_bans

    if (columns_emb["roles_pick"]):
        dim += emb_size['roles'] * 2 * n_players
    if (columns_emb["roles_ban"]):
        dim += emb_size['roles'] * 2 * n_bans
    if (columns_emb["attributes"]):
        dim += emb_size['attributes'] * 2 * n_players
    if (columns_emb["attack"]):
        dim += emb_size['attack'] * 2 * n_players
    if (columns_emb["items"]):
        dim += emb_size['items'] * 2 * n_players
    if (columns_emb["backpack"]):
        dim += emb_size['item'] * 2 * n_players
    if (columns_emb["neutral"]):
        dim += emb_size['neutral'] * 2 * n_players
    return dim