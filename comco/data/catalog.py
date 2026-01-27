from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def read_tsv(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            rows.append([p.strip() for p in ln.split("\t")])
    return rows


def load_images(image_root: str, imageid_file: str) -> Tuple[List[str], Dict[str, str]]:
    rows = read_tsv(imageid_file)
    img_ids: List[str] = []
    img_paths: Dict[str, str] = {}
    for parts in rows:
        if len(parts) < 2:
            continue
        image_id, folder_name = parts[0], parts[1]
        folder = os.path.join(image_root, folder_name)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        if not files:
            continue
        img_path = os.path.join(folder, files[0])
        if not os.path.isfile(img_path):
            continue
        img_ids.append(image_id)
        img_paths[image_id] = img_path
    return img_ids, img_paths


def load_entities(entities_full_file: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    rows = read_tsv(entities_full_file)
    text_ids: List[str] = []
    descs: Dict[str, str] = {}
    names: Dict[str, str] = {}
    for parts in rows:
        tid = None
        name = ""
        desc = ""
        if len(parts) >= 4:
            tid = parts[0]
            desc = parts[3]
            if len(parts) >= 3:
                name = parts[-2]
                desc = parts[-1]
        elif len(parts) >= 3:
            tid = parts[0]
            name = parts[-2]
            desc = parts[-1]
        elif len(parts) >= 2:
            tid = parts[0]
            desc = parts[1]
        if tid is None:
            continue
        text_ids.append(tid)
        descs[tid] = desc
        if name:
            names[tid] = name
    return text_ids, descs, names


@dataclass
class DatasetCatalog:
    dataset_root: str
    image_root: str
    imageid_file: str
    entities_full_file: str

    @classmethod
    def from_args(
        cls,
        dataset_root: str,
        image_root: str,
        imageid_file: str = "imageid.txt",
        entities_full_file: str = "entities_full.txt",
    ) -> "DatasetCatalog":
        return cls(
            dataset_root=dataset_root,
            image_root=image_root,
            imageid_file=os.path.join(dataset_root, imageid_file),
            entities_full_file=os.path.join(dataset_root, entities_full_file),
        )

    def validate(self) -> None:
        for p in [self.imageid_file, self.entities_full_file]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(self.image_root)

    def load(self) -> Tuple[List[str], Dict[str, str], List[str], Dict[str, str], Dict[str, str]]:
        self.validate()
        image_ids, image_paths = load_images(self.image_root, self.imageid_file)
        text_ids, text_descs, text_names = load_entities(self.entities_full_file)
        return image_ids, image_paths, text_ids, text_descs, text_names
