#!/usr/bin/env python3
"""
prompt_versioning.py — A simple prompt versioning system.

Stores prompts as named entries, each with a full history of versions.
Every save creates a new immutable version; nothing is overwritten.

Storage: a single JSON file (default: prompts.json) acting as a
lightweight local database. Swap out PromptStore's _load/_save methods
if you want a real DB backend later.

Usage as a library:
    from prompt_versioning import PromptStore

    store = PromptStore("prompts.json")
    store.create("greeting", "You are a helpful assistant.", author="alice")
    store.update("greeting", "You are a helpful, concise assistant.", author="alice", message="tightened tone")
    latest = store.get("greeting")              # latest version
    v1 = store.get("greeting", version=1)        # specific version
    store.diff("greeting", 1, 2)                 # unified diff between versions
    store.rollback("greeting", 1, author="alice") # creates a new version copying v1
    store.tag("greeting", 2, "production")
    store.get("greeting", tag="production")
    store.history("greeting")

CLI:
    python prompt_versioning.py create greeting "You are a helpful assistant." --author alice
    python prompt_versioning.py update greeting "You are a helpful, concise assistant." --author alice -m "tightened tone"
    python prompt_versioning.py get greeting
    python prompt_versioning.py get greeting --version 1
    python prompt_versioning.py history greeting
    python prompt_versioning.py diff greeting 1 2
    python prompt_versioning.py rollback greeting 1 --author alice
    python prompt_versioning.py tag greeting 2 production
    python prompt_versioning.py list
"""

import argparse
import difflib
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class PromptVersion:
    version: int
    content: str
    author: str
    message: str
    created_at: str
    tags: List[str] = field(default_factory=list)


@dataclass
class PromptEntry:
    name: str
    versions: List[PromptVersion] = field(default_factory=list)

    def latest(self) -> PromptVersion:
        return self.versions[-1]

    def get_version(self, version: int) -> Optional[PromptVersion]:
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_by_tag(self, tag: str) -> Optional[PromptVersion]:
        for v in reversed(self.versions):
            if tag in v.tags:
                return v
        return None


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

class PromptNotFoundError(Exception):
    pass


class VersionNotFoundError(Exception):
    pass


class PromptStore:
    def __init__(self, path: str = "prompts.json"):
        self.path = Path(path)
        self._data: Dict[str, PromptEntry] = {}
        self._load()

    # ---- persistence ----

    def _load(self):
        if not self.path.exists():
            self._data = {}
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._data = {}
        for name, entry in raw.items():
            versions = [PromptVersion(**v) for v in entry["versions"]]
            self._data[name] = PromptEntry(name=name, versions=versions)

    def _save(self):
        serializable = {
            name: {"name": entry.name,
                   "versions": [asdict(v) for v in entry.versions]}
            for name, entry in self._data.items()
        }
        self.path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    # ---- core operations ----

    def create(self, name: str, content: str, author: str = "", message: str = "initial version") -> PromptVersion:
        if name in self._data:
            raise ValueError(f"Prompt '{name}' already exists. Use update() to add a new version.")
        version = PromptVersion(
            version=1,
            content=content,
            author=author,
            message=message,
            created_at=_now(),
        )
        self._data[name] = PromptEntry(name=name, versions=[version])
        self._save()
        return version

    def update(self, name: str, content: str, author: str = "", message: str = "") -> PromptVersion:
        entry = self._require(name)
        next_num = entry.latest().version + 1
        version = PromptVersion(
            version=next_num,
            content=content,
            author=author,
            message=message,
            created_at=_now(),
        )
        entry.versions.append(version)
        self._save()
        return version

    def get(self, name: str, version: Optional[int] = None, tag: Optional[str] = None) -> PromptVersion:
        entry = self._require(name)
        if tag is not None:
            v = entry.get_by_tag(tag)
            if v is None:
                raise VersionNotFoundError(f"No version of '{name}' tagged '{tag}'")
            return v
        if version is None:
            return entry.latest()
        v = entry.get_version(version)
        if v is None:
            raise VersionNotFoundError(f"'{name}' has no version {version}")
        return v

    def history(self, name: str) -> List[PromptVersion]:
        entry = self._require(name)
        return list(entry.versions)

    def diff(self, name: str, version_a: int, version_b: int) -> str:
        entry = self._require(name)
        a = entry.get_version(version_a)
        b = entry.get_version(version_b)
        if a is None or b is None:
            raise VersionNotFoundError(f"Versions {version_a} and/or {version_b} not found for '{name}'")
        diff_lines = difflib.unified_diff(
            a.content.splitlines(keepends=True),
            b.content.splitlines(keepends=True),
            fromfile=f"{name} v{version_a}",
            tofile=f"{name} v{version_b}",
        )
        return "".join(diff_lines)

    def rollback(self, name: str, to_version: int, author: str = "", message: Optional[str] = None) -> PromptVersion:
        entry = self._require(name)
        target = entry.get_version(to_version)
        if target is None:
            raise VersionNotFoundError(f"'{name}' has no version {to_version}")
        msg = message or f"rollback to v{to_version}"
        return self.update(name, target.content, author=author, message=msg)

    def tag(self, name: str, version: int, tag: str):
        entry = self._require(name)
        v = entry.get_version(version)
        if v is None:
            raise VersionNotFoundError(f"'{name}' has no version {version}")
        # Remove this tag from any other version of the same prompt
        # so a tag like "production" always points to exactly one version.
        for other in entry.versions:
            if tag in other.tags and other is not v:
                other.tags.remove(tag)
        if tag not in v.tags:
            v.tags.append(tag)
        self._save()

    def list_prompts(self) -> List[str]:
        return sorted(self._data.keys())

    def delete(self, name: str):
        self._require(name)
        del self._data[name]
        self._save()

    # ---- helpers ----

    def _require(self, name: str) -> PromptEntry:
        if name not in self._data:
            raise PromptNotFoundError(f"No prompt named '{name}'")
        return self._data[name]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simple prompt versioning system.")
    parser.add_argument("--store", default="prompts.json", help="Path to the JSON store file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new prompt (version 1)")
    p_create.add_argument("name")
    p_create.add_argument("content")
    p_create.add_argument("--author", default="")
    p_create.add_argument("-m", "--message", default="initial version")

    p_update = sub.add_parser("update", help="Add a new version to an existing prompt")
    p_update.add_argument("name")
    p_update.add_argument("content")
    p_update.add_argument("--author", default="")
    p_update.add_argument("-m", "--message", default="")

    p_get = sub.add_parser("get", help="Get a prompt (latest, by version, or by tag)")
    p_get.add_argument("name")
    p_get.add_argument("--version", type=int, default=None)
    p_get.add_argument("--tag", default=None)

    p_hist = sub.add_parser("history", help="Show version history for a prompt")
    p_hist.add_argument("name")

    p_diff = sub.add_parser("diff", help="Diff two versions of a prompt")
    p_diff.add_argument("name")
    p_diff.add_argument("version_a", type=int)
    p_diff.add_argument("version_b", type=int)

    p_rollback = sub.add_parser("rollback", help="Roll back to an earlier version (creates a new version)")
    p_rollback.add_argument("name")
    p_rollback.add_argument("to_version", type=int)
    p_rollback.add_argument("--author", default="")

    p_tag = sub.add_parser("tag", help="Tag a specific version (e.g. 'production')")
    p_tag.add_argument("name")
    p_tag.add_argument("version", type=int)
    p_tag.add_argument("tag_name")

    sub.add_parser("list", help="List all prompt names")

    p_delete = sub.add_parser("delete", help="Delete a prompt and all its versions")
    p_delete.add_argument("name")

    args = parser.parse_args()
    store = PromptStore(args.store)

    try:
        if args.command == "create":
            v = store.create(args.name, args.content, author=args.author, message=args.message)
            print(f"Created '{args.name}' v{v.version}")

        elif args.command == "update":
            v = store.update(args.name, args.content, author=args.author, message=args.message)
            print(f"Saved '{args.name}' v{v.version}")

        elif args.command == "get":
            v = store.get(args.name, version=args.version, tag=args.tag)
            print(f"--- {args.name} v{v.version} ({v.created_at}, by {v.author or 'unknown'}) ---")
            if v.tags:
                print(f"tags: {', '.join(v.tags)}")
            print(v.content)

        elif args.command == "history":
            for v in store.history(args.name):
                tags = f" [{', '.join(v.tags)}]" if v.tags else ""
                print(f"v{v.version}  {v.created_at}  {v.author or 'unknown':<12} {v.message}{tags}")

        elif args.command == "diff":
            d = store.diff(args.name, args.version_a, args.version_b)
            print(d if d else "(no differences)")

        elif args.command == "rollback":
            v = store.rollback(args.name, args.to_version, author=args.author)
            print(f"Rolled back '{args.name}' to content of v{args.to_version} as new v{v.version}")

        elif args.command == "tag":
            store.tag(args.name, args.version, args.tag_name)
            print(f"Tagged '{args.name}' v{args.version} as '{args.tag_name}'")

        elif args.command == "list":
            names = store.list_prompts()
            if not names:
                print("(no prompts yet)")
            for n in names:
                print(n)

        elif args.command == "delete":
            store.delete(args.name)
            print(f"Deleted '{args.name}'")

    except (PromptNotFoundError, VersionNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()