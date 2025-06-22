from typing import Dict, List, Tuple
from dota.dataset import Dota2Dataset


class Dota2Schema:
    """
    Schema para dados de partidas de Dota 2.
    Fornece configuração de colunas, embeddings e cálculo de dimensão de entrada.
    """

    def __init__(self, dataset: Dota2Dataset, n_players: int = 5, n_bans: int = 7, n_items: int = 6, n_backpack: int = 3, n_neutral: int = 1):
        # Configuração das colunas do dataset
        self.dota_columns: Dict[str, Dict[str, bool]] = {
            "match_id": {'include': False, 'normalize': False},
            'player_slot': {'include': False, 'normalize': False},
            "start_time": {'include': False, 'normalize': False},
            "obs_placed": {'include': True, 'normalize': True},
            "sen_placed": {'include': True, 'normalize': True},
            "creeps_stacked": {'include': True, 'normalize': True},
            "camps_stacked": {'include': True, 'normalize': True},
            "rune_pickups": {'include': True, 'normalize': True},
            "firstblood_claimed": {'include': False, 'normalize': False},
            "teamfight_participation": {'include': True, 'normalize': False},
            "towers_killed": {'include': True, 'normalize': True},
            "roshans_killed": {'include': True, 'normalize': True},
            "stuns": {'include': True, 'normalize': True},
            "times": {'include': False, 'normalize': False},
            "gold_t": {'include': False, 'normalize': False},
            "lh_t": {'include': False, 'normalize': False},
            "dn_t": {'include': False, 'normalize': False},
            "xp_t": {'include': False, 'normalize': False},
            "party_id": {'include': False, 'normalize': False},
            "account_id": {'include': False, 'normalize': False},
            "hero_id": {'include': True, 'normalize': False},
            "items_vector": {'include': False, 'normalize': False},
            "backpack_vector": {'include': False, 'normalize': False},
            "item_neutral": {'include': False, 'normalize': False},
            "kills": {'include': True, 'normalize': True},
            "deaths": {'include': True, 'normalize': True},
            "assists": {'include': True, 'normalize': True},
            "last_hits": {'include': True, 'normalize': True},
            "denies": {'include': True, 'normalize': True},
            "gold_per_min": {'include': True, 'normalize': True},
            "xp_per_min": {'include': True, 'normalize': True},
            "level": {'include': True, 'normalize': True},
            "net_worth": {'include': True, 'normalize': True},
            "hero_damage": {'include': True, 'normalize': True},
            "tower_damage": {'include': True, 'normalize': True},
            "hero_healing": {'include': True, 'normalize': True},
            "gold": {'include': True, 'normalize': True},
            "gold_spent": {'include': True, 'normalize': True},
            "neutral_kills": {'include': True, 'normalize': True},
            "tower_kills": {'include': True, 'normalize': True},
            "courier_kills": {'include': True, 'normalize': True},
            "lane_kills": {'include': True, 'normalize': True},
            "hero_kills": {'include': True, 'normalize': True},
            "observer_kills": {'include': True, 'normalize': True},
            "sentry_kills": {'include': True, 'normalize': True},
            "roshan_kills": {'include': True, 'normalize': True},
            "necronomicon_kills": {'include': True, 'normalize': True},
            "ancient_kills": {'include': True, 'normalize': True},
            "buyback_count": {'include': True, 'normalize': True},
            "purchase_ward_observer": {'include': True, 'normalize': True},
            "purchase_ward_sentry": {'include': True, 'normalize': True},
            "purchase_gem": {'include': True, 'normalize': True},
            "purchase_rapier": {'include': True, 'normalize': True},
            "is_pick": {'include': True, 'normalize': False},
            "team": {'include': True, 'normalize': False},
            "order": {'include': True, 'normalize': False},
            "version": {'include': False, 'normalize': True},
            "leagueid": {'include': False, 'normalize': True},
            "start_date_time": {'include': False, 'normalize': True},
            "duration": {'include': False, 'normalize': True},
            "patch": {'include': False, 'normalize': True},
            "region": {'include': False, 'normalize': True},
            "series_id": {'include': False, 'normalize': True},
            "series_type": {'include': False, 'normalize': True},
            "radiant_win": {'include': False, 'normalize': True},
            "tower_status_radiant": {'include': False, 'normalize': True},
            "tower_status_dire": {'include': False, 'normalize': True},
            "barracks_status_radiant": {'include': False, 'normalize': True},
            "barracks_status_dire": {'include': False, 'normalize': True},
            "leaguename": {'include': False, 'normalize': True},
            "tier": {'include': False, 'normalize': True},
            "gold_adv": {'include': False, 'normalize': True},
            "exp_adv": {'include': False, 'normalize': True},
            "hero_name": {'include': False, 'normalize': True},
            "attack_type": {'include': False, 'normalize': True},
            "roles_vector": {'include': False, 'normalize': True},
            "attack_range": {'include': False, 'normalize': True},
            "move_speed": {'include': False, 'normalize': True},
            "day_vision": {'include': False, 'normalize': True},
            "night_vision": {'include': False, 'normalize': True},
        }
        self.columns_emb: Dict[str, bool] = {
            "picks": True,
            "bans": True,
            "roles": True,
            "attributes": True,
            "attack": True,
            "items": True,
            "backpack": True,
            "neutral": True,
        }
        self._included_fields_count = sum(
            1 for v in self.dota_columns.values() if v['include'])

        # Número de jogadores e bans
        self.n_players: int = n_players
        self.n_bans: int = n_bans

        # Itens e configurações de mochila/neutro
        self.n_items: int = n_items
        self.n_backpack: int = n_backpack
        self.n_neutral: int = n_neutral

        # Dimensões padrão dos embeddings
        self.emb_dim_config: Dict[str, int] = {
            'emb_pick_dim': 32,
            'emb_ban_dim': 32,
            'emb_item_dim': 16,
            'emb_role_dim': 16,
            'emb_attr_dim': 8,
            'emb_neutral_dim': 8,
        }
        # Listas de índices para embeddings
        self.emb_config: Dict[str, List[int]] = {
            'emb_attr': list(dataset.config.attr_mapping.values()),
            'emb_role': list(dataset.config.role_mapping.values()),
            'emb_item': list(dataset.config.item_mapping.values()),
            'emb_hero': list(dataset.config.hero_mapping.values()),
        }

        self.embedding: Dict[str, Tuple[bool, List[int],
                                        int]] = self.get_embedding_configuration()
        self.input: int = self.calculate_input_dimension(
            n_players=self.n_players,
            n_bans=self.n_bans,
            emb_size={k: len(v) for k, v in self.emb_config.items()}
        )

    def get_embedding_configuration(
        self,
    ) -> Dict[str, Tuple[bool, List[int], int]]:
        """
        Retorna configuração de embeddings para cada coluna relevante de forma flexível e escalável.
        """
        c = self.columns_emb
        config = self.emb_config
        dim = self.emb_dim_config

        # Mapeamento: nome da feature -> (flag, chave_config, chave_dim)
        emb_map = {
            "radiant_picks":      (c['picks'],     'emb_hero', 'emb_pick_dim'),
            "dire_picks":         (c['picks'],     'emb_hero', 'emb_pick_dim'),
            "radiant_bans":       (c['bans'],      'emb_hero', 'emb_ban_dim'),
            "dire_bans":          (c['bans'],      'emb_hero', 'emb_ban_dim'),
            "radiant_roles_picks": (c['roles'],     'emb_role', 'emb_role_dim'),
            "dire_roles_picks":   (c['roles'],     'emb_role', 'emb_role_dim'),
            "radiant_roles_bans": (c['roles'],     'emb_role', 'emb_role_dim'),
            "dire_roles_bans":    (c['roles'],     'emb_role', 'emb_role_dim'),
            "radiant_attributes": (c['attributes'], 'emb_attr', 'emb_attr_dim'),
            "dire_attributes":    (c['attributes'], 'emb_attr', 'emb_attr_dim'),
            "radiant_attack":     (c['attack'],    'emb_attr', 'emb_attr_dim'),
            "dire_attack":        (c['attack'],    'emb_attr', 'emb_attr_dim'),
            "radiant_items":      (c['items'],     'emb_item', 'emb_item_dim'),
            "dire_items":         (c['items'],     'emb_item', 'emb_item_dim'),
            "radiant_backpack":   (c['backpack'],  'emb_item', 'emb_item_dim'),
            "dire_backpack":      (c['backpack'],  'emb_item', 'emb_item_dim'),
            "radiant_neutral":    (c['neutral'],   'emb_item', 'emb_neutral_dim'),
            "dire_neutral":       (c['neutral'],   'emb_item', 'emb_neutral_dim'),
        }
        result = {}
        for key, (flag, conf_key, dim_key) in emb_map.items():
            result[key] = (flag, config[conf_key], dim[dim_key])
        return result

    def calculate_input_dimension(
        self,
        n_players: int,
        n_bans: int,
        emb_size: Dict[str, int]
    ) -> int:
        """
        Calcula a dimensão de entrada do modelo.
        """
        c = self.columns_emb
        dim = 2 * (n_players + n_bans) * self._included_fields_count
        emb_fields = [
            (c['picks'], 'pick', 2 * n_players),
            (c['bans'], 'ban', 2 * n_bans),
            (c.get('roles'), 'roles', 2 * n_players),
            (c.get('attributes'), 'attributes', 2 * n_players),
            (c.get('attack'), 'attack', 2 * n_players),
            (c.get('items'), 'items', 2 * n_players),
            (c.get('backpack'), 'item', 2 * n_players),
            (c.get('neutral'), 'neutral', 2 * n_players),
        ]
        for flag, key, scale in emb_fields:
            if flag:
                if key not in emb_size:
                    raise KeyError(
                        f"Chave '{key}' não encontrada em emb_size.")
                dim += emb_size[key] * scale
        return dim
