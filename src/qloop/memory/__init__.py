"""Phase-2A/2B `LSA-NL` Split-Horizon Continuum Memory primitives (`KVMemBlob`, `DLMG`, `ESM`)."""
from __future__ import annotations

from .esm import (
    DEFAULT_CORE_LOOP_SPEC,
    EpisodicSideMemoryBank,
    ESMConfig,
    ESMLedgerEntry,
    NullSpaceESMAnchor,
    dgks_orthonormalize_rows,
    project_orthogonal_dgks,
)
from .kv_blob import (
    HEADER_METADATA_TAIL_BYTES,
    DynamicLatentMemoryGraph,
    KVBlobConfig,
    KVBlobHeader,
    KVMemBlob,
    SplitHorizonKVBlobStore,
    mount_kv_blob,
    pack_kv_blob,
    strip_sinks_and_unrotate_k,
    theoretical_h64_int4_bytes,
    theoretical_header_wire_bytes,
)

__all__ = [
    "DEFAULT_CORE_LOOP_SPEC",
    "HEADER_METADATA_TAIL_BYTES",
    "DynamicLatentMemoryGraph",
    "ESMConfig",
    "ESMLedgerEntry",
    "EpisodicSideMemoryBank",
    "KVBlobConfig",
    "KVBlobHeader",
    "KVMemBlob",
    "NullSpaceESMAnchor",
    "SplitHorizonKVBlobStore",
    "dgks_orthonormalize_rows",
    "mount_kv_blob",
    "pack_kv_blob",
    "project_orthogonal_dgks",
    "strip_sinks_and_unrotate_k",
    "theoretical_h64_int4_bytes",
    "theoretical_header_wire_bytes",
]
