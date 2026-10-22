"""
core/atlas_gameologist.py

sends to:
gameologist/

Gameologist export module for Centroids.

Rules:
- Output Leiden clusters at resolution 1.5 (fallback: HDBSCAN Cluster 1)
- Output semantic and burrows relationships between CHANNELS
- Hash channel IDs (SHA-256, first 16 hex) — used as opaque keys for
  cluster/relationship files and per-channel filenames.
- XOR-obfuscate channel names, channel handles, and video IDs with a key
  that also lives in gameologist.js. This is obfuscation, not encryption:
  its purpose is to delete the single bulk-lookup file
  (reverse_lookup.json) and force any scraper to actually read/execute
  the client JS. It is not a security boundary against a determined
  attacker.
- Filter: Gaming category only, >= 8 minutes, never live, posted after
  2020-01-01.
- No titles, URLs, timestamps, or transcripts in gameologist metadata.

Channel search policy
---------------------
The client searches by CHANNEL HANDLE or CHANNEL ID only. Display names
are intentionally not matched, because display names collide between
channels (e.g. multiple "Ctop"-ish display names). The exporter therefore
stores both:
    "name"   — display name (for messages only, XOR-encoded)
    "handle" — channel handle without "@" (used for search, XOR-encoded)

Output layout (matches index.html / gameologist.js):

    gameologist/
      index.html
      gameologist.js
      data/
        manifest.json              # {version, data_version, generated, shards}
        shard_000.json, ...        # {hashed_channel_id: {id, name, handle, v: [[xor_id, dur, YYYYMMDD], ...]}}
        channel_clusters.json      # {hashed_channel_id: int}
        burrows_channel_clusters.json  # {hashed_channel_id: int}
        semantic_relationships.json# {hashed_channel_id: [[hashed_channel_id, sim], ...]}
        burrows_relationships.json # {hashed_channel_id: [[hashed_channel_id, sim], ...]}

NOTE: The client hashes the user's channel-ID input to match hashed keys,
XOR-decodes channel handles for handle matching, XOR-decodes channel names
for display, and XOR-decodes video IDs at click time.

Size optimizations:
  1. Relationship lists are capped at MAX_RELATIONSHIPS_PER_CHANNEL.
  2. Relationships are positional arrays [hash, similarity], rounded.
  3. Per-channel video entries are positional [xor_id, dur, YYYYMMDD_int].
  4. All JSON is compact (separators=(",", ":")).
  5. Channel-ID hash format unchanged: SHA-256, first 16 hex chars, same salt.
"""

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_HASH_SALT = "im_not_a_fartologist"

# XOR key for channel names, channel handles, and video IDs. MUST match
# XOR_KEY in gameologist.js exactly. Obfuscation only — the key is public.
XOR_KEY = 'im_not_a_fartologist_fart_pad_7c3a9f2e'

TARGET_LEIDEN_RESOLUTION = 1.3
HDBSCAN_CLUSTER_1_KEYS = ("hdbscan_cluster_1", "hdbscan", "cluster_1", "cluster")

MAX_RELATIONSHIPS_PER_CHANNEL = 100
SIMILARITY_DECIMALS = 3

TARGET_SHARDS = 40


def _dump_compact(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def xor_encode(value: str, key: str = XOR_KEY) -> str:
    """XOR-encode a UTF-8 string with a repeating key; return lowercase hex."""
    if value is None:
        value = ""
    key_bytes = key.encode("utf-8")
    data_bytes = str(value).encode("utf-8")
    out = bytearray(len(data_bytes))
    for i, b in enumerate(data_bytes):
        out[i] = b ^ key_bytes[i % len(key_bytes)]
    return out.hex()


def _extract_handle(video_meta: Dict, folder_name: str) -> str:
    _UC_ID = re.compile(r"UC[A-Za-z0-9_-]{22}")

    def _looks_like_channel_id(s: str) -> bool:
        return bool(s) and bool(_UC_ID.fullmatch(s))

    # 1. @handle embedded in a URL — most authoritative
    for field in ("uploader_url", "channel_url"):
        val = video_meta.get(field)
        if not val:
            continue
        m = re.search(r"@([^/?#]+)", str(val))
        if m:
            return m.group(1).strip()

    # 2. uploader_id with @ prefix (modern yt-dlp)
    val = video_meta.get("uploader_id")
    if val:
        s = str(val).strip()
        if s.startswith("@"):
            s = s[1:]
            if s and not _looks_like_channel_id(s):
                return s

    # 3. Folder name (Centroids input convention)
    if folder_name:
        s = folder_name.lstrip("@").strip()
        if s and not _looks_like_channel_id(s):
            return s

    # 4. Explicit handle fields (rare)
    for field in ("channel_handle", "uploader_handle", "handle"):
        val = video_meta.get(field)
        if val:
            s = str(val).strip().lstrip("@")
            if s and not _looks_like_channel_id(s):
                return s

    # 5. LAST RESORT — display name. Known to collide between channels;
    #    only reached when no real handle exists.
    val = video_meta.get("channel_name")
    if val:
        s = str(val).strip().lstrip("@")
        if s and not _looks_like_channel_id(s):
            return s

    return ""

class GameologistExporter:
    """Exports Centroids data for the gameologist sister project."""

    def __init__(self, centroids_dir: Path, gameologist_dir: Path):
        self.centroids_dir = Path(centroids_dir)
        self.gameologist_dir = Path(gameologist_dir)
        self.gameologist_data_dir = self.gameologist_dir / "data"
        self.gameologist_data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_string(value: str, salt: str = DEFAULT_HASH_SALT) -> str:
        """SHA-256(value+salt), first 16 hex chars (matches gameologist.js)."""
        combined = (value or "") + salt
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Channel Sharding
    # ------------------------------------------------------------------
    def _write_channel_shards(self, channels: List[Dict]) -> List[str]:
        """Write channels into ~TARGET_SHARDS bundle files. Returns shard filenames."""
        if not channels:
            return []
        shard_size = max(1, math.ceil(len(channels) / TARGET_SHARDS))
        shard_files: List[str] = []
        for i in range(0, len(channels), shard_size):
            chunk = channels[i:i + shard_size]
            shard = {c["id"]: c for c in chunk}
            name = f"shard_{i // shard_size:03d}.json"
            _dump_compact(self.gameologist_data_dir / name, shard)
            shard_files.append(name)
        print(f"  Wrote {len(shard_files)} shard files (avg {len(channels) // len(shard_files)} channels per shard)")
        return shard_files

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def _write_manifest(self, shard_files: List[str]) -> None:
        manifest = {
            "version": "1.1",
            "data_version": "1.1",
            "generated": datetime.now().isoformat(),
            "shards": sorted(shard_files),
        }
        _dump_compact(self.gameologist_data_dir / "manifest.json", manifest)
        print(f"  Wrote manifest with {len(shard_files)} shard files")

    # ------------------------------------------------------------------
    # Input loading + filtering
    # ------------------------------------------------------------------
    def load_channel_metadata(self, channel_name: str) -> Optional[List[Dict]]:
        path = self.centroids_dir / "data" / "input" / channel_name / "metadata.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else None
        except Exception:
            return None

    def filter_video_metadata(self, video_meta: Dict) -> bool:
        if video_meta.get("category") != "Gaming":
            return False
        duration = video_meta.get("duration") or 0
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            return False
        if duration < 480:
            return False
        if video_meta.get("was_live") or video_meta.get("is_live"):
            return False
        upload_date = video_meta.get("upload_date", "")
        if not isinstance(upload_date, str) or len(upload_date) != 8:
            return False
        try:
            video_date = datetime(
                int(upload_date[:4]), int(upload_date[4:6]), int(upload_date[6:8])
            )
        except (ValueError, IndexError):
            return False
        return video_date >= datetime(2020, 1, 1)

    # ------------------------------------------------------------------
    # Per-channel export
    # ------------------------------------------------------------------
    def export_channel_metadata(self, channel_name: str) -> Optional[Dict]:
        """
        Export a single channel's filtered, hashed/obfuscated metadata.
        Returns None if the channel has no videos surviving the filter.

        Output schema:
            {
              "id":     "<hashed_channel_id>",     # SHA-256[:16] of raw channel ID
              "name":   "<xor_hex_channel_name>",  # XOR hex of raw display name
              "handle": "<xor_hex_channel_handle>",# XOR hex of handle (no leading @)
              "v":      [[<xor_hex_video_id>, <duration>, <YYYYMMDD int>], ...]
            }
        """
        metadata = self.load_channel_metadata(channel_name)
        if not metadata:
            return None

        filtered_videos = [v for v in metadata if self.filter_video_metadata(v)]
        if not filtered_videos:
            return None

        first_video = filtered_videos[0]
        channel_id_raw = first_video.get("channel_id", "") or ""
        channel_name_raw = first_video.get("channel_name", "") or ""
        channel_handle_raw = _extract_handle(first_video, channel_name)

        hashed_channel_id = self.hash_string(channel_id_raw)
        encoded_channel_name = xor_encode(channel_name_raw)
        encoded_channel_handle = xor_encode(channel_handle_raw)

        video_entries: List[List[Any]] = []
        for video in filtered_videos:
            video_id_raw = video.get("id", "") or ""
            encoded_video_id = xor_encode(video_id_raw)

            try:
                upload_date_int = int(video.get("upload_date", "0") or 0)
            except (TypeError, ValueError):
                upload_date_int = 0

            try:
                duration_int = int(video.get("duration") or 0)
            except (TypeError, ValueError):
                duration_int = 0

            video_entries.append([
                encoded_video_id,
                duration_int,
                upload_date_int,
            ])

        return {
            "id": hashed_channel_id,
            "name": encoded_channel_name,
            "handle": encoded_channel_handle,
            "v": video_entries,
        }

    # ------------------------------------------------------------------
    # Cluster variant selection
    # ------------------------------------------------------------------
    @staticmethod
    def _resolution_from_label(label: Any) -> Optional[float]:
        if not label:
            return None
        m = re.search(r"r\s*=\s*([\d.]+)", str(label))
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _select_cluster_variant(
        self,
        default_labels: Optional[List[Optional[int]]],
        cluster_method: str,
        cluster_variant_labels: Optional[Dict[str, List[Optional[int]]]],
        cluster_metric_labels: Optional[Dict[str, str]],
    ):
        if not cluster_variant_labels:
            return default_labels, f"default labels ({cluster_method})"

        metric_labels = cluster_metric_labels or {}

        best_key = None
        best_diff = float('inf')
        best_res = None
        for key in cluster_variant_labels:
            res = self._resolution_from_label(metric_labels.get(key))
            if res is None:
                continue
            diff = abs(res - TARGET_LEIDEN_RESOLUTION)
            if diff < best_diff:
                best_diff = diff
                best_key = key
                best_res = res

        if best_key is not None:
            desc = metric_labels.get(best_key, f"Leiden r={best_res:.2f}")
            return cluster_variant_labels[best_key], desc

        for key in HDBSCAN_CLUSTER_1_KEYS:
            if key in cluster_variant_labels:
                desc = metric_labels.get(key, key)
                return cluster_variant_labels[key], desc

        first_key = next(iter(cluster_variant_labels))
        desc = metric_labels.get(first_key, f"first available variant ({first_key})")
        return cluster_variant_labels[first_key], desc

    # ------------------------------------------------------------------
    # Cluster export
    # ------------------------------------------------------------------
    def export_cluster_assignments(
        self,
        cluster_labels: Optional[List[Optional[int]]],
        texts: List[str],
        cluster_method: str = "hdbscan",
        leiden_resolution: float = 1.0,
        cluster_variant_labels: Optional[Dict[str, List[Optional[int]]]] = None,
        folder_to_hashed_id: Optional[Dict[str, str]] = None,
        cluster_metric_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        chosen_labels, description = self._select_cluster_variant(
            default_labels=cluster_labels,
            cluster_method=cluster_method,
            cluster_variant_labels=cluster_variant_labels,
            cluster_metric_labels=cluster_metric_labels,
        )
        print(f"  Cluster source: {description}")

        channel_clusters: Dict[str, int] = {}
        if chosen_labels is not None:
            for text, label in zip(texts, chosen_labels):
                if label is None:
                    continue
                if folder_to_hashed_id and text in folder_to_hashed_id:
                    hashed_id = folder_to_hashed_id[text]
                else:
                    hashed_id = self.hash_string(text)
                channel_clusters[hashed_id] = int(label)
        else:
            print("  WARNING: no cluster labels available; writing empty cluster file")

        _dump_compact(
            self.gameologist_data_dir / "channel_clusters.json",
            channel_clusters,
        )
        return channel_clusters

    def export_burrows_cluster_assignments(
        self,
        cluster_variant_labels: Optional[Dict[str, List[Optional[int]]]],
        texts: List[str],
        folder_to_hashed_id: Optional[Dict[str, str]] = None,
        cluster_metric_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        if not cluster_variant_labels:
            return {}

        burrows_keys = [
            key for key in cluster_variant_labels
            if key == "burrows_cluster" or key.startswith("burrows_cluster_")
        ]
        if not burrows_keys:
            return {}

        metric_labels = cluster_metric_labels or {}
        selected_key = burrows_keys[0]
        best_diff = float("inf")
        for key in burrows_keys:
            resolution = self._resolution_from_label(metric_labels.get(key))
            if resolution is None:
                continue
            difference = abs(resolution - TARGET_LEIDEN_RESOLUTION)
            if difference < best_diff:
                best_diff = difference
                selected_key = key

        labels = cluster_variant_labels[selected_key]
        channel_clusters: Dict[str, int] = {}
        for text, label in zip(texts, labels):
            if label is None:
                continue
            hashed_id = (
                folder_to_hashed_id.get(text)
                if folder_to_hashed_id else None
            ) or self.hash_string(text)
            channel_clusters[hashed_id] = int(label)

        _dump_compact(
            self.gameologist_data_dir / "burrows_channel_clusters.json",
            channel_clusters,
        )
        desc = metric_labels.get(selected_key, selected_key)
        print(f"  Burrows cluster source: {desc}")
        return channel_clusters

    # ------------------------------------------------------------------
    # Relationship export
    # ------------------------------------------------------------------
    @staticmethod
    def _build_relationships(
        similarity_matrix: np.ndarray,
        texts: List[str],
        folder_to_hashed_id: Dict[str, str],
        hash_fn,
    ) -> Dict[str, List[List[Any]]]:
        relationships: Dict[str, List[List[Any]]] = {}
        n = len(texts)
        cap = MAX_RELATIONSHIPS_PER_CHANNEL

        for i in range(n):
            source_text = texts[i]
            source_hashed = folder_to_hashed_id.get(source_text) or hash_fn(source_text)

            sims = np.array(similarity_matrix[i], dtype=float, copy=True)
            sims[i] = -1.0
            order = np.argsort(sims)[::-1]

            entries: List[List[Any]] = []
            for idx in order:
                if len(entries) >= cap:
                    break
                score = float(sims[idx])
                if score <= 0.0:
                    break
                if not np.isfinite(score):
                    continue
                target_text = texts[int(idx)]
                target_hashed = (
                    folder_to_hashed_id.get(target_text) or hash_fn(target_text)
                )
                entries.append([
                    target_hashed,
                    round(score, SIMILARITY_DECIMALS),
                ])

            relationships[source_hashed] = entries

        return relationships

    def export_relationships(
        self,
        similarity_matrix: np.ndarray,
        burrows_similarity_matrix: np.ndarray,
        texts: List[str],
        folder_to_hashed_id: Dict[str, str],
    ) -> Dict[str, Any]:
        semantic = self._build_relationships(
            similarity_matrix, texts, folder_to_hashed_id, self.hash_string
        )
        burrows = self._build_relationships(
            burrows_similarity_matrix, texts, folder_to_hashed_id, self.hash_string
        )

        _dump_compact(
            self.gameologist_data_dir / "semantic_relationships.json",
            semantic,
        )
        _dump_compact(
            self.gameologist_data_dir / "burrows_relationships.json",
            burrows,
        )

        print(f"  Wrote semantic relationships for {len(semantic)} channels "
              f"(max {MAX_RELATIONSHIPS_PER_CHANNEL} per channel)")
        print(f"  Wrote burrows relationships for {len(burrows)} channels "
              f"(max {MAX_RELATIONSHIPS_PER_CHANNEL} per channel)")

        return {"semantic": semantic, "burrows": burrows}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def export_all(
        self,
        texts: List[str],
        cluster_labels: Optional[List[Optional[int]]],
        similarity_matrix: np.ndarray,
        burrows_similarity_matrix: np.ndarray,
        cluster_method: str = "hdbscan",
        leiden_resolution: float = 1.0,
        cluster_variant_labels: Optional[Dict[str, List[Optional[int]]]] = None,
        cluster_metric_labels: Optional[Dict[str, str]] = None,
        burrows_cluster_variant_labels: Optional[Dict[str, List[Optional[int]]]] = None,
        burrows_cluster_metric_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        print("\nExporting data for gameologist...")

        exported_channels: Dict[str, Dict] = {}
        folder_to_hashed_id: Dict[str, str] = {}

        for text in texts:
            result = self.export_channel_metadata(text)
            if result:
                exported_channels[text] = result
                folder_to_hashed_id[text] = result["id"]

        if not exported_channels:
            print("  WARNING: no channels passed the gameologist filter")
        else:
            print(f"  Exported metadata for {len(exported_channels)} channels")

        clusters = self.export_cluster_assignments(
            cluster_labels=cluster_labels,
            texts=texts,
            cluster_method=cluster_method,
            leiden_resolution=leiden_resolution,
            cluster_variant_labels=cluster_variant_labels,
            folder_to_hashed_id=folder_to_hashed_id,
            cluster_metric_labels=cluster_metric_labels,
        )
        print(f"  Exported cluster assignments for {len(clusters)} channels")
        burrows_clusters = self.export_burrows_cluster_assignments(
            cluster_variant_labels=burrows_cluster_variant_labels,
            texts=texts,
            folder_to_hashed_id=folder_to_hashed_id,
            cluster_metric_labels=burrows_cluster_metric_labels,
        )
        print(f"  Exported Burrows cluster assignments for {len(burrows_clusters)} channels")

        relationships = self.export_relationships(
            similarity_matrix, burrows_similarity_matrix, texts, folder_to_hashed_id
        )

        shard_files = self._write_channel_shards(list(exported_channels.values()))
        self._write_manifest(shard_files)

        print("  Gameologist export complete!")
        return {
            "channels": exported_channels,
            "clusters": clusters,
            "burrows_clusters": burrows_clusters,
            "relationships": relationships,
        }