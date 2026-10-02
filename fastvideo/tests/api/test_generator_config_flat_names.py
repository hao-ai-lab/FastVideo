# SPDX-License-Identifier: Apache-2.0
"""Every ``GeneratorConfig`` field must have a defined place in the flat keyword API.

``fastvideo.api.compat`` converts between ``GeneratorConfig`` and the flat keywords of
``VideoGenerator.from_pretrained`` / ``FastVideoArgs`` through each field's ``flat_name`` metadata. A field without a
flat name must be listed in ``_SPECIALLY_MAPPED_PATHS``, so a new field cannot be dropped silently by the conversion.
"""
from fastvideo.api.compat import _FLAT_NAME_FIELDS, _SPECIALLY_MAPPED_PATHS, _schema_fields
from fastvideo.api.schema import FLAT_NAME, GeneratorConfig


def test_every_field_has_a_flat_name_or_a_special_mapping() -> None:
    unmapped = [
        dotted_path for dotted_path, config_field, _ in _schema_fields(GeneratorConfig)
        if FLAT_NAME not in config_field.metadata and dotted_path not in _SPECIALLY_MAPPED_PATHS
    ]
    assert unmapped == []


def test_flat_names_are_unique() -> None:
    flat_names = [
        config_field.metadata[FLAT_NAME] for _, config_field, _ in _schema_fields(GeneratorConfig)
        if FLAT_NAME in config_field.metadata
    ]
    assert len(flat_names) == len(set(flat_names)) == len(_FLAT_NAME_FIELDS)


def test_special_mappings_name_existing_fields() -> None:
    field_paths = {dotted_path for dotted_path, _, _ in _schema_fields(GeneratorConfig)}
    assert set(_SPECIALLY_MAPPED_PATHS) <= field_paths
