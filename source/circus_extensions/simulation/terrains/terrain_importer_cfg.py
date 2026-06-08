# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab import terrains as isaac_terrains
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter


@configclass
class TerrainImporterCfg(isaac_terrains.TerrainImporterCfg):
    """Configuration for the terrain manager."""

    class_type: type = TerrainImporter
    """The class to use for the terrain importer.

    Defaults to :class:`omni.isaac.lab.terrains.terrain_importer.TerrainImporter`.
    """