"""Low-level fantasy scoring engine (stat rows to total_pts)."""

import pandas as pd
from typing import Dict, Optional, Iterable


class ScoringEngine:
    """
    Robust scoring utilities that support both your DEFAULT_SCORING_RULES and NEW_SCORING_RULES.
    Works on nflverse weekly player stats (plus merged kicking) and optional team-defense frames.
    """

    # --- public API ----------------------------------------------------------

    @staticmethod
    def prepare_offense_kicking_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Idempotently creates/repairs composite columns needed by either rule set.
        Safe to call multiple times. Missing source columns are treated as 0.
        """
        df = df.copy()

        # Helpers
        def get_stat_or_default(col, default=0):
            return df[col].fillna(default) if col in df.columns else default

        # Yardage bonuses (default rules)
        df['passing_yards_300_399_game'] = (
            df.get('passing_yards', 0).between(300, 399).astype(int)
            if 'passing_yards' in df.columns else 0
        )
        df['passing_yards_400_plus_game'] = (
            (df.get('passing_yards', 0) >= 400).astype(int)
            if 'passing_yards' in df.columns else 0
        )

        df['receiving_yards_100_199_game'] = (
            df.get('receiving_yards', 0).between(100, 199).astype(int)
            if 'receiving_yards' in df.columns else 0
        )
        df['receiving_yards_200_plus_game'] = (
            (df.get('receiving_yards', 0) >= 200).astype(int)
            if 'receiving_yards' in df.columns else 0
        )

        df['rushing_yards_100_199_game'] = (
            df.get('rushing_yards', 0).between(100, 199).astype(int)
            if 'rushing_yards' in df.columns else 0
        )
        df['rushing_yards_200_plus_game'] = (
            (df.get('rushing_yards', 0) >= 200).astype(int)
            if 'rushing_yards' in df.columns else 0
        )

        # Total fumbles lost (default rules)
        df['total_fumbles_lost'] = (
            get_stat_or_default('sack_fumbles_lost') + get_stat_or_default('rushing_fumbles_lost') + get_stat_or_default('receiving_fumbles_lost')
        )

        # Normalize synonyms (so rules can use either)
        ScoringEngine._normalize_offense_synonyms(df)

        # Kicking composites:
        # 1) FG buckets for default rules
        if not {'fg_made_0_19','fg_made_20_29','fg_made_30_39'}.issubset(df.columns):
            # If the bucket counts aren't in the file, try to derive from fg_made_list distances if present.
            ScoringEngine._derive_fg_buckets_from_lists(df)

        # Sum 0-39 bucket (your code relied on this)
        df['fg_made_0_39'] = (
            get_stat_or_default('fg_made_0_19') + get_stat_or_default('fg_made_20_29') + get_stat_or_default('fg_made_30_39')
        )

        # 2) FG made yardage for new rules (per-yard scoring)
        if 'fg_made_yards' not in df.columns:
            df['fg_made_yards'] = ScoringEngine._compute_total_made_fg_yards(df)

        # PAT/XPs normalization
        if 'xp_made' not in df.columns and 'pat_made' in df.columns:
            df['xp_made'] = df['pat_made']
        if 'xp_missed' not in df.columns and 'pat_missed' in df.columns:
            df['xp_missed'] = df['pat_missed']

        # Two-point conversions unified (new rules key)
        df['two_point_conversions'] = (
            get_stat_or_default('passing_2pt_conversions') + get_stat_or_default('rushing_2pt_conversions') + get_stat_or_default('receiving_2pt_conversions')
        )

        # Touchdown synonyms (default uses *_tds; new uses *_touchdowns)
        if 'passing_touchdowns' not in df.columns and 'passing_tds' in df.columns:
            df['passing_touchdowns'] = df['passing_tds']
        if 'rushing_touchdowns' not in df.columns and 'rushing_tds' in df.columns:
            df['rushing_touchdowns'] = df['rushing_tds']
        if 'receiving_touchdowns' not in df.columns and 'receiving_tds' in df.columns:
            df['receiving_touchdowns'] = df['receiving_tds']

        return df

    @staticmethod
    def apply_scoring(df: pd.DataFrame, scoring_rules: Dict[str, Optional[float]]) -> pd.DataFrame:
        """
        Applies scoring rules to an offense/kicker/player frame that has been run through
        prepare_offense_kicking_features(). Skips any rule whose inputs are missing.
        Adds/overwrites column: 'total_pts'.
        """
        df = df.copy()
        # make sure composites/synonyms exist
        df = ScoringEngine.prepare_offense_kicking_features(df)

        # Normalize provided rule keys to avoid double counting across synonyms
        rules = ScoringEngine._normalize_rule_weights(scoring_rules)

        print(f"Rules: {rules}")

        # Start fresh and guarantee numeric dtype
        df['total_pts'] = 0.0

        # Linear keys that map 1:1 to df columns
        linear_map = {
            # Passing
            "passing_yards": "passing_yards",
            "passing_tds": "passing_tds",
            "passing_touchdowns": "passing_touchdowns",
            "interceptions": "interceptions",
            "passing_2pt_conversions": "passing_2pt_conversions",
            "two_point_conversions": "two_point_conversions",

            # Rushing
            "rushing_yards": "rushing_yards",
            "rushing_tds": "rushing_tds",
            "rushing_touchdowns": "rushing_touchdowns",
            "rushing_2pt_conversions": "rushing_2pt_conversions",

            # Receiving
            "receptions": "receptions",
            "receiving_yards": "receiving_yards",
            "receiving_tds": "receiving_tds",
            "receiving_touchdowns": "receiving_touchdowns",
            "receiving_2pt_conversions": "receiving_2pt_conversions",

            # Fumbles
            "total_fumbles_lost": "total_fumbles_lost",
            "sack_fumbles_lost": "sack_fumbles_lost",
            "rushing_fumbles_lost": "rushing_fumbles_lost",
            "receiving_fumbles_lost": "receiving_fumbles_lost",

            # Kicking (flat/yardage)
            "fg_missed": "fg_missed",
            "fg_made_yards": "fg_made_yards",
            "xp_made": "xp_made",
            "xp_missed": "xp_missed",
            "pat_made": "xp_made",     # synonyms
            "pat_missed": "xp_missed",

            # Special Teams player TDs (from player table if present)
            "special_teams_tds": "special_teams_tds",
        }

        # Coerce only the columns that will be used to numeric for robustness
        used_columns: set[str] = set()
        for rule_key, weight in rules.items():
            if weight is None:
                continue
            col = linear_map.get(rule_key, rule_key)
            if col in df.columns:
                used_columns.add(col)
        # Bonus columns (applied later)
        bonus_map = {
            'passing_yards_300_399_game': 'passing_yards_300_399_game',
            'passing_yards_400_plus_game': 'passing_yards_400_plus_game',
            'receiving_yards_100_199_game': 'receiving_yards_100_199_game',
            'receiving_yards_200_plus_game': 'receiving_yards_200_plus_game',
            'rushing_yards_100_199_game': 'rushing_yards_100_199_game',
            'rushing_yards_200_plus_game': 'rushing_yards_200_plus_game',
        }
        for rule_key, col in bonus_map.items():
            if rule_key in rules and col in df.columns:
                used_columns.add(col)
        # FG tiered buckets (default rules)
        fg_bucket_map = {
            "fg_made_0_39": "fg_made_0_39",
            "fg_made_40_49": "fg_made_40_49",
            "fg_made_50_59": "fg_made_50_59",
            "fg_made_60_": "fg_made_60_",
        }
        for rule_key, col in fg_bucket_map.items():
            if rule_key in rules and col in df.columns:
                used_columns.add(col)
        # Yardage FG
        if 'fg_made_yards' in rules and 'fg_made_yards' in df.columns:
            used_columns.add('fg_made_yards')

        ScoringEngine._coerce_numeric(df, list(used_columns))

        # Apply linear rules
        for rule_key, weight in rules.items():
            if weight is None:
                continue  # non-linear (e.g., points_allowed tiers)
            col = linear_map.get(rule_key, rule_key)  # try synonym, else literal
            if col in df.columns:
                df['total_pts'] += df[col].fillna(0) * weight

        # Yardage bonus rules (default) — must be applied after base scoring
        for rule_key, col in bonus_map.items():
            if rule_key in rules and col in df.columns:
                df['total_pts'] += df[col].fillna(0) * rules[rule_key]

        # FG tiered buckets (default rules)
        # You defined: fg_made_0_39, fg_made_40_49, fg_made_50_59, fg_made_60_
        for rule_key, col in fg_bucket_map.items():
            if rule_key in rules and col in df.columns:
                df['total_pts'] += df[col].fillna(0) * rules[rule_key]

        # Ensure numeric and no NaNs
        df['total_pts'] = pd.to_numeric(df['total_pts'], errors='coerce').fillna(0.0)

        return df

    @staticmethod
    def apply_team_def_scoring(def_df: pd.DataFrame, scoring_rules: Dict[str, Optional[float]]) -> pd.DataFrame:
        """
        Applies team defense scoring if `def_df` has the needed columns. Adds/overwrites 'def_total_pts'.
        Expected cols (if present): points_allowed, def_td, sacks, def_interception, def_fumble_recovery,
        safeties, def_forced_fumble, def_blocked_kick, st_def_td, st_def_forced_fumble, st_def_fumble_recovery.
        """
        df = def_df.copy()
        df['def_total_pts'] = 0.0

        # Build normalized weights to avoid double counting across synonyms
        rules = dict(scoring_rules or {})

        # Create missing composites/synonyms if present in feed
        if 'st_def_td' not in df.columns:
            # Try to derive from explicit KR/PR TD columns if available
            kr = df.get('kick_return_touchdowns')
            pr = df.get('punt_return_touchdowns')
            if kr is not None or pr is not None:
                df['st_def_td'] = (kr.fillna(0) if kr is not None else 0) + (pr.fillna(0) if pr is not None else 0)

        # Prefer generic keys if provided; else fall back to legacy keys
        def _get_weight(*keys: str) -> Optional[float]:
            for k in keys:
                if k in rules and rules[k] is not None:
                    return rules[k]
            return None

        # Map of (preferred_keys...) -> source column in df
        td_specs = [
            (("def_touchdowns", "def_td"), "def_td"),
            (("sacks", "def_sack"), "sacks"),
            (("def_interceptions", "def_interception"), "def_interception"),
            (("def_fumbles_recovered", "def_fumble_recovery"), "def_fumble_recovery"),
            (("safeties", "def_safety"), "safeties"),
            (("def_forced_fumble",), "def_forced_fumble"),
            (("blocked_kicks", "def_blocked_kick"), "def_blocked_kick"),
            # Special teams / returns
            (("kick_return_touchdowns",), "kick_return_touchdowns"),
            (("punt_return_touchdowns",), "punt_return_touchdowns"),
            (("st_def_td",), "st_def_td"),
            (("st_def_forced_fumble",), "st_def_forced_fumble"),
            (("st_def_fumble_recovery",), "st_def_fumble_recovery"),
        ]

        # If explicit KR/PR TD weights are provided, ignore aggregate st_def_td weight to avoid double counting
        st_td_weight_blocked = any(k in rules and rules[k] is not None for k in ("kick_return_touchdowns", "punt_return_touchdowns"))

        used_numeric_cols: list[str] = []
        for keys, src_col in td_specs:
            if src_col not in df.columns:
                continue
            if src_col == 'st_def_td' and st_td_weight_blocked:
                continue
            w = _get_weight(*keys)
            if w is None:
                continue
            used_numeric_cols.append(src_col)
        ScoringEngine._coerce_numeric(df, used_numeric_cols)

        for keys, src_col in td_specs:
            if src_col not in df.columns:
                continue
            if src_col == 'st_def_td' and st_td_weight_blocked:
                continue
            w = _get_weight(*keys)
            if w is None:
                continue
            df['def_total_pts'] += df[src_col].fillna(0) * w

        # Points allowed tiers (non-linear)
        if 'points_allowed' in df.columns and 'def_points_allowed_0' in scoring_rules:
            df['def_total_pts'] += ScoringEngine._score_points_allowed_tiers(
                df['points_allowed'],
                scoring_rules
            )

        return df

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _normalize_offense_synonyms(df: pd.DataFrame) -> None:
        """Create common synonyms if only one naming style exists."""
        # INTs exist in both rule sets as 'interceptions' → nothing to map.
        # touchdowns: *_tds ⇄ *_touchdowns handled in prepare function.
        # PAT vs XP handled in prepare function.
        # 2PT unified handled in prepare function.
        pass

    # --- helpers for normalization / coercion --------------------------------

    @staticmethod
    def _normalize_rule_weights(scoring_rules: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Returns a copy of scoring_rules with mutually-exclusive synonyms resolved to a single key
        to prevent double counting. Preference order:
        - touchdowns: *_touchdowns over *_tds
        - XP vs PAT: xp_* over pat_*
        - FG: if fg_made_yards present, ignore bucket rules
        - Fumbles: if any specific fumbles lost provided, ignore total_fumbles_lost
        """
        if not scoring_rules:
            return {}
        rules = dict(scoring_rules)

        # Touchdowns groups
        td_groups = [
            ("passing_touchdowns", "passing_tds"),
            ("rushing_touchdowns", "rushing_tds"),
            ("receiving_touchdowns", "receiving_tds"),
        ]
        for preferred, alias in td_groups:
            if preferred in rules and rules.get(preferred) is not None:
                # Drop alias to avoid double counting
                rules.pop(alias, None)
            elif alias in rules and rules.get(alias) is not None:
                # Keep alias, it's fine
                pass

        # XP vs PAT
        xp_groups = [("xp_made", "pat_made"), ("xp_missed", "pat_missed")]
        for preferred, alias in xp_groups:
            if preferred in rules and rules.get(preferred) is not None:
                rules.pop(alias, None)

        # FG: yards vs buckets
        if rules.get('fg_made_yards') is not None:
            for k in ("fg_made_0_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_"):
                rules.pop(k, None)

        # Fumbles: prefer specific components over total
        if any(rules.get(k) is not None for k in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")):
            rules.pop("total_fumbles_lost", None)

        return rules

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
        """Coerces selected columns to numeric, in-place, filling NaNs with 0.0 for safe arithmetic."""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- lightweight, unit-testable examples ---------------------------------

    @staticmethod
    def example_offense_scoring_row() -> tuple[pd.DataFrame, float]:
        """
        Returns a single-row DataFrame and its computed total under a sample rule set.

        This is intended for unit tests; consumers can assert the returned total.

        Example:
            df, total = ScoringEngine.example_offense_scoring_row()
            assert round(total, 2) == round(df['total_pts'].iloc[0], 2)
        """
        sample = pd.DataFrame([
            dict(
                passing_yards=305, passing_tds=2, interceptions=1,
                rushing_yards=20, rushing_tds=0,
                receptions=5, receiving_yards=50, receiving_tds=1,
            )
        ])
        rules = {
            # Passing
            "passing_yards": 0.04,
            "passing_tds": 4,
            "interceptions": -2,
            # Bonuses
            'passing_yards_300_399_game': 2,
            'passing_yards_400_plus_game': 6,
            # Rushing
            "rushing_yards": 0.1,
            "rushing_tds": 6,
            # Receiving
            "receptions": 1,
            "receiving_yards": 0.1,
            "receiving_tds": 6,
        }
        out = ScoringEngine.apply_scoring(sample, rules)
        return out, float(out['total_pts'].iloc[0])

    @staticmethod
    def example_kicking_scoring_row() -> tuple[pd.DataFrame, float]:
        """
        Returns a single-row DataFrame for kickers and computed total using yardage-based rules.
        """
        sample = pd.DataFrame([
            dict(
                fg_made_list='53;44;28', fg_missed=1, xp_made=3, xp_missed=0,
            )
        ])
        rules = {
            "fg_made_yards": 0.1,  # 53+44+28 = 125 yards → 12.5 pts
            "fg_missed": -1,
            "xp_made": 1,
            "xp_missed": -1,
        }
        out = ScoringEngine.apply_scoring(sample, rules)
        return out, float(out['total_pts'].iloc[0])

    @staticmethod
    def _derive_fg_buckets_from_lists(df: pd.DataFrame) -> None:
        """
        Populate fg_made_0_19 / 20_29 / 30_39 / 40_49 / 50_59 / 60_ if fg_made_list exists.
        nflverse kicking has 'fg_made_list' like '53;44;28' and 'fg_missed_list' similarly.
        """
        if 'fg_made_list' not in df.columns:
            # Nothing to do
            return

        def _parse_to_ints(s: pd.Series) -> Iterable[list]:
            vals = []
            for x in s.fillna(''):
                if not x:
                    vals.append([])
                else:
                    # lists separated by ; or ,
                    parts = [p for p in str(x).replace(',', ';').split(';') if p.strip() != '']
                    vals.append([ScoringEngine._safe_int(p) for p in parts if ScoringEngine._safe_int(p) is not None])
            return vals

        made_lists = _parse_to_ints(df['fg_made_list'])
        buckets = {'0_19': [], '20_29': [], '30_39': [], '40_49': [], '50_59': [], '60_': []}

        for lst in made_lists:
            counts = {'0_19':0, '20_29':0, '30_39':0, '40_49':0, '50_59':0, '60_':0}
            for d in lst:
                if d <= 19: counts['0_19'] += 1
                elif d <= 29: counts['20_29'] += 1
                elif d <= 39: counts['30_39'] += 1
                elif d <= 49: counts['40_49'] += 1
                elif d <= 59: counts['50_59'] += 1
                else: counts['60_'] += 1
            for k in counts:
                buckets[k].append(counts[k])

        # create columns if missing
        for k in ['0_19','20_29','30_39','40_49','50_59','60_']:
            col = f'fg_made_{k}'
            if col not in df.columns:
                df[col] = pd.Series(buckets[k], index=df.index)

    @staticmethod
    def _compute_total_made_fg_yards(df: pd.DataFrame) -> pd.Series:
        """
        Return per-row sum of made FG distances.
        Prefer 'fg_made_distance' if a numeric series exists, otherwise sum parsed 'fg_made_list'.
        """
        if 'fg_made_distance' in df.columns:
            # Some feeds store the sum-of-distances directly; ensure numeric
            s = pd.to_numeric(df['fg_made_distance'], errors='coerce').fillna(0)
            return s

        if 'fg_made_list' in df.columns:
            out = []
            for x in df['fg_made_list'].fillna(''):
                if not x:
                    out.append(0)
                    continue
                parts = [p for p in str(x).replace(',', ';').split(';') if p.strip() != '']
                dists = [ScoringEngine._safe_int(p) or 0 for p in parts]
                out.append(sum(dists))
            return pd.Series(out, index=df.index)

        # fallback: try to reconstruct from buckets if present
        buckets = ['fg_made_0_19','fg_made_20_29','fg_made_30_39','fg_made_40_49','fg_made_50_59','fg_made_60_']
        if set(buckets).issubset(df.columns):
            # use midpoints for an approximation (better than 0)
            mid = {'fg_made_0_19':10, 'fg_made_20_29':25, 'fg_made_30_39':35,
                   'fg_made_40_49':45, 'fg_made_50_59':55, 'fg_made_60_':62}
            approx = sum(df[b]*mid[b] for b in buckets)
            return approx.fillna(0)

        return pd.Series(0, index=df.index)

    @staticmethod
    def _safe_int(x) -> Optional[int]:
        try:
            return int(str(x).strip())
        except Exception:
            return None

    @staticmethod
    def _score_points_allowed_tiers(points_allowed: pd.Series, rules: Dict[str, float]) -> pd.Series:
        """
        Score tiered 'points allowed' per your new rules if present.
        """
        pa = points_allowed.fillna(0)
        s = pd.Series(0.0, index=pa.index)

        def add(mask, key):
            if key in rules:
                s[mask] += rules[key]

        add(pa == 0, "def_points_allowed_0")
        add((pa >= 1) & (pa <= 6), "def_points_allowed_1_6")
        add((pa >= 7) & (pa <= 13), "def_points_allowed_7_13")
        add((pa >= 14) & (pa <= 20), "def_points_allowed_14_20")
        # 21-27: neutral (0) — nothing to add
        add((pa >= 28) & (pa <= 34), "def_points_allowed_28_34")
        add(pa >= 35, "def_points_allowed_35_plus")

        return s