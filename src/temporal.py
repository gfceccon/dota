# import ast
# import polars as pl
# from src.dataset import players

# gold_t = (
#     players
#     .select("match_id", "player_account_id", "player_gold_t")
#     .group_by(["player_account_id", "match_id"])
#     .agg(pl.col("player_gold_t").first())
#     .select(
#         pl.col("player_gold_t").map_elements(
#             lambda x: ast.literal_eval(x) if x and isinstance(x, str) else [],
#             return_dtype=pl.List(pl.Int64)
#         ).alias("gold_t_list")
#     )
#     .select(
#         pl.col("gold_t_list").list.len().alias("gold_t_length"),
#     )
#     .group_by("gold_t_length")
#     .agg(pl.col("gold_t_length").count().alias("count"))
#     .filter(pl.col("gold_t_length") > 20)
#     .sort("gold_t_length", descending=True)
#     .collect()
# )


# gold_t = [
#     {'gold_t_length': 113, 'count': 10},
#     {'gold_t_length': 110, 'count': 10},
#     {'gold_t_length': 109, 'count': 10},
#     {'gold_t_length': 108, 'count': 20},
#     {'gold_t_length': 105, 'count': 10},
#     {'gold_t_length': 101, 'count': 10},
#     {'gold_t_length': 99, 'count': 10},
#     {'gold_t_length': 97, 'count': 10},
#     {'gold_t_length': 96, 'count': 10},
#     {'gold_t_length': 95, 'count': 20},
#     {'gold_t_length': 93, 'count': 10},
#     {'gold_t_length': 92, 'count': 20},
#     {'gold_t_length': 91, 'count': 10},
#     {'gold_t_length': 90, 'count': 20},
#     {'gold_t_length': 89, 'count': 20},
#     {'gold_t_length': 88, 'count': 20},
#     {'gold_t_length': 87, 'count': 30},
#     {'gold_t_length': 86, 'count': 40},
#     {'gold_t_length': 85, 'count': 20},
#     {'gold_t_length': 84, 'count': 20},
#     {'gold_t_length': 83, 'count': 10},
#     {'gold_t_length': 82, 'count': 40},
#     {'gold_t_length': 81, 'count': 50},
#     {'gold_t_length': 80, 'count': 10},
#     {'gold_t_length': 79, 'count': 50},
#     {'gold_t_length': 78, 'count': 60},
#     {'gold_t_length': 77, 'count': 60},
#     {'gold_t_length': 76, 'count': 60},
#     {'gold_t_length': 75, 'count': 80},
#     {'gold_t_length': 74, 'count': 100},
#     {'gold_t_length': 73, 'count': 170},
#     {'gold_t_length': 72, 'count': 140},
#     {'gold_t_length': 71, 'count': 80},
#     {'gold_t_length': 70, 'count': 130},
#     {'gold_t_length': 69, 'count': 140},
#     {'gold_t_length': 68, 'count': 250},
#     {'gold_t_length': 67, 'count': 380},
#     {'gold_t_length': 66, 'count': 210},
#     {'gold_t_length': 65, 'count': 360},
#     {'gold_t_length': 64, 'count': 290},
#     {'gold_t_length': 63, 'count': 380},
#     {'gold_t_length': 62, 'count': 370},
#     {'gold_t_length': 61, 'count': 450},
#     {'gold_t_length': 60, 'count': 620},
#     {'gold_t_length': 59, 'count': 500},
#     {'gold_t_length': 58, 'count': 780},
#     {'gold_t_length': 57, 'count': 740},
#     {'gold_t_length': 56, 'count': 920},
#     {'gold_t_length': 55, 'count': 1200},
#     {'gold_t_length': 54, 'count': 1280},
#     {'gold_t_length': 53, 'count': 1478},
#     {'gold_t_length': 52, 'count': 1949},
#     {'gold_t_length': 51, 'count': 2310},
#     {'gold_t_length': 50, 'count': 2443},
#     {'gold_t_length': 49, 'count': 2520},
#     {'gold_t_length': 48, 'count': 2680},
#     {'gold_t_length': 47, 'count': 3549},
#     {'gold_t_length': 46, 'count': 3170},
#     {'gold_t_length': 45, 'count': 4560},
#     {'gold_t_length': 44, 'count': 4960},
#     {'gold_t_length': 43, 'count': 5940},
#     {'gold_t_length': 42, 'count': 6320},
#     {'gold_t_length': 41, 'count': 6224},
#     {'gold_t_length': 40, 'count': 6750},
#     {'gold_t_length': 39, 'count': 7470},
#     {'gold_t_length': 38, 'count': 8530},
#     {'gold_t_length': 37, 'count': 9700},
#     {'gold_t_length': 36, 'count': 10430},
#     {'gold_t_length': 35, 'count': 10920},
#     {'gold_t_length': 34, 'count': 13388},
#     {'gold_t_length': 33, 'count': 14448},
#     {'gold_t_length': 32, 'count': 16060},
#     {'gold_t_length': 31, 'count': 15640},
#     {'gold_t_length': 30, 'count': 15570},
#     {'gold_t_length': 29, 'count': 15380},
#     {'gold_t_length': 28, 'count': 15740},
#     {'gold_t_length': 27, 'count': 15330},
#     {'gold_t_length': 26, 'count': 13820},
#     {'gold_t_length': 25, 'count': 11190},
#     {'gold_t_length': 24, 'count': 8752},
#     {'gold_t_length': 23, 'count': 6930},
#     {'gold_t_length': 22, 'count': 5762},
#     {'gold_t_length': 21, 'count': 5692}
# ]
