#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Append a training run record into an xlsx workbook without extra deps."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"

ET.register_namespace("", MAIN_NS)
ET.register_namespace("r", DOC_REL_NS)


def _col_to_num(col: str) -> int:
    value = 0
    for ch in col:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value


def _num_to_col(index: int) -> str:
    letters: list[str] = []
    value = index
    while value > 0:
        value, rem = divmod(value - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def _cell_ref_parts(ref: str) -> tuple[str, int]:
    col = "".join(ch for ch in ref if ch.isalpha())
    row = "".join(ch for ch in ref if ch.isdigit())
    return col, int(row)


def _load_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(zf.read(name))


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = _load_xml(zf, "xl/sharedStrings.xml")
    strings: list[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        texts = [node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t")]
        strings.append("".join(texts))
    return strings


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "s":
        value = cell.findtext(f"{{{MAIN_NS}}}v", default="")
        if value == "":
            return ""
        return shared_strings[int(value)]
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(f".//{{{MAIN_NS}}}t")]
        return "".join(texts)
    return cell.findtext(f"{{{MAIN_NS}}}v", default="")


def _sheet_xml_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = _load_xml(zf, "xl/workbook.xml")
    rel_id = None
    for sheet in workbook.findall(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet"):
        if sheet.get("name") == sheet_name:
            rel_id = sheet.get(f"{{{DOC_REL_NS}}}id")
            break
    if rel_id is None:
        raise ValueError(f"Worksheet not found: {sheet_name}")

    rels = _load_xml(zf, "xl/_rels/workbook.xml.rels")
    target = None
    for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if rel.get("Id") == rel_id:
            target = rel.get("Target")
            break
    if target is None:
        raise ValueError(f"Worksheet relationship not found: {sheet_name}")

    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join("xl", target))


def _find_headers(sheet_root: ET.Element, shared_strings: list[str]) -> list[tuple[str, str]]:
    sheet_data = sheet_root.find(f"{{{MAIN_NS}}}sheetData")
    if sheet_data is None:
        raise ValueError("sheetData missing from workbook")

    header_row = None
    for row in sheet_data.findall(f"{{{MAIN_NS}}}row"):
        if row.get("r") == "1":
            header_row = row
            break
    if header_row is None:
        raise ValueError("Header row missing from worksheet")

    headers: list[tuple[str, str]] = []
    for cell in header_row.findall(f"{{{MAIN_NS}}}c"):
        ref = cell.get("r", "")
        col, _ = _cell_ref_parts(ref)
        value = _cell_text(cell, shared_strings).strip()
        if not value:
            break
        headers.append((col, value))
    if not headers:
        raise ValueError("No headers found in worksheet")
    return headers


def _style_map(sheet_root: ET.Element) -> dict[str, str]:
    sheet_data = sheet_root.find(f"{{{MAIN_NS}}}sheetData")
    if sheet_data is None:
        return {}

    rows = sheet_data.findall(f"{{{MAIN_NS}}}row")
    for row in reversed(rows):
        if row.get("r") == "1":
            continue
        styles: dict[str, str] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            style = cell.get("s")
            if not style:
                continue
            col, _ = _cell_ref_parts(cell.get("r", ""))
            styles[col] = style
        if styles:
            return styles
    return {}


def _next_row_number(sheet_root: ET.Element) -> int:
    sheet_data = sheet_root.find(f"{{{MAIN_NS}}}sheetData")
    if sheet_data is None:
        return 2
    row_numbers = [
        int(row.get("r"))
        for row in sheet_data.findall(f"{{{MAIN_NS}}}row")
        if row.get("r")
    ]
    return (max(row_numbers) if row_numbers else 1) + 1


def _build_cell(ref: str, value: Any, style: str | None) -> ET.Element:
    cell = ET.Element(f"{{{MAIN_NS}}}c", {"r": ref})
    if style is not None:
        cell.set("s", style)

    if isinstance(value, bool):
        numeric = "1" if value else "0"
        ET.SubElement(cell, f"{{{MAIN_NS}}}v").text = numeric
        return cell

    if isinstance(value, (int, float)):
        ET.SubElement(cell, f"{{{MAIN_NS}}}v").text = str(value)
        return cell

    text = str(value)
    cell.set("t", "inlineStr")
    inline = ET.SubElement(cell, f"{{{MAIN_NS}}}is")
    text_node = ET.SubElement(inline, f"{{{MAIN_NS}}}t")
    if text != text.strip():
        text_node.set(f"{{{XML_NS}}}space", "preserve")
    text_node.text = text
    return cell


def _update_dimension(sheet_root: ET.Element, last_col: str, last_row: int) -> None:
    dimension = sheet_root.find(f"{{{MAIN_NS}}}dimension")
    if dimension is None:
        dimension = ET.Element(f"{{{MAIN_NS}}}dimension")
        sheet_root.insert(1, dimension)
    dimension.set("ref", f"A1:{last_col}{last_row}")


def _append_row(
    sheet_root: ET.Element,
    record: dict[str, Any],
    headers: list[tuple[str, str]],
) -> int:
    sheet_data = sheet_root.find(f"{{{MAIN_NS}}}sheetData")
    if sheet_data is None:
        raise ValueError("sheetData missing from workbook")

    target_row = _next_row_number(sheet_root)
    style_map = _style_map(sheet_root)

    row = ET.Element(
        f"{{{MAIN_NS}}}row",
        {
            "r": str(target_row),
            "spans": f"1:{len(headers)}",
        },
    )

    for col, header in headers:
        value = record.get(header)
        if value in (None, ""):
            continue
        ref = f"{col}{target_row}"
        row.append(_build_cell(ref, value, style_map.get(col)))

    sheet_data.append(row)
    _update_dimension(sheet_root, headers[-1][0], target_row)
    return target_row


def _write_workbook(
    workbook_path: Path,
    sheet_path: str,
    updated_xml: bytes,
) -> None:
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=workbook_path.parent,
        prefix=f".{workbook_path.name}.",
        suffix=".tmp",
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(workbook_path, "r") as src, zipfile.ZipFile(
            temp_path,
            "w",
        ) as dst:
            for item in src.infolist():
                data = updated_xml if item.filename == sheet_path else src.read(item.filename)
                dst.writestr(item, data)
        os.replace(temp_path, workbook_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a run record to xlsx.")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--sheet", default="train_log")
    args = parser.parse_args()

    workbook_path = Path(args.workbook)
    with open(args.record, encoding="utf-8") as f:
        record = json.load(f)
    if not isinstance(record, dict):
        raise ValueError("Record JSON must be an object")

    with zipfile.ZipFile(workbook_path, "r") as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_path = _sheet_xml_path(zf, args.sheet)
        sheet_root = _load_xml(zf, sheet_path)
        headers = _find_headers(sheet_root, shared_strings)
        row_number = _append_row(sheet_root, record, headers)
        updated_xml = ET.tostring(
            sheet_root,
            encoding="utf-8",
            xml_declaration=True,
        )

    _write_workbook(workbook_path, sheet_path, updated_xml)
    print(
        json.dumps(
            {
                "workbook": str(workbook_path),
                "sheet": args.sheet,
                "row": row_number,
                "columns": len(headers),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
