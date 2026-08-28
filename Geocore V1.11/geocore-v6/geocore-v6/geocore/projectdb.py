"""
Project database (SQLite).

Replaces v4's scattered hardcoded paths with a real store: projects, runs,
configs, metrics, and artifact paths - full provenance so any result a
client sees can be traced to the exact config and data that produced it.
SQLite is the right call for a desktop product (zero-ops, single file,
ACID); the schema maps 1:1 onto PostgreSQL/PostGIS when you go multi-user.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    commodity TEXT NOT NULL,
    created_at REAL NOT NULL,
    notes TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running',
    config_json TEXT NOT NULL,
    engine_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (run_id, kind)
);
"""


class ProjectDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- projects ---
    def create_project(self, name: str, commodity: str, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO projects (name, commodity, created_at, notes) "
            "VALUES (?, ?, ?, ?)", (name, commodity, time.time(), notes))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_project(self, name: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT id, name, commodity, created_at, notes FROM projects "
            "WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return dict(zip(("id", "name", "commodity", "created_at", "notes"), row))

    def list_projects(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT id, name, commodity, created_at FROM projects "
            "ORDER BY created_at DESC").fetchall()
        return [dict(zip(("id", "name", "commodity", "created_at"), r))
                for r in rows]

    # --- runs ---
    def start_run(self, project_id: int, config_dict: dict,
                  engine_version: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (project_id, started_at, config_json, "
            "engine_version) VALUES (?, ?, ?, ?)",
            (project_id, time.time(), json.dumps(config_dict), engine_version))
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "complete") -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ? WHERE id = ?",
            (time.time(), status, run_id))
        self.conn.commit()

    def log_metric(self, run_id: int, key: str, value: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO metrics (run_id, key, value) "
            "VALUES (?, ?, ?)", (run_id, key, float(value)))
        self.conn.commit()

    def log_artifact(self, run_id: int, kind: str, path: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO artifacts (run_id, kind, path) "
            "VALUES (?, ?, ?)", (run_id, kind, str(path)))
        self.conn.commit()

    def run_summary(self, run_id: int) -> Dict:
        run = self.conn.execute(
            "SELECT id, project_id, started_at, finished_at, status, "
            "config_json, engine_version FROM runs WHERE id = ?",
            (run_id,)).fetchone()
        metrics = dict(self.conn.execute(
            "SELECT key, value FROM metrics WHERE run_id = ?",
            (run_id,)).fetchall())
        artifacts = dict(self.conn.execute(
            "SELECT kind, path FROM artifacts WHERE run_id = ?",
            (run_id,)).fetchall())
        return {
            "run": dict(zip(("id", "project_id", "started_at", "finished_at",
                             "status", "config_json", "engine_version"), run)),
            "metrics": metrics,
            "artifacts": artifacts,
        }

    def list_runs(self) -> List[Dict]:
        """Flat run history for UI display (newest first)."""
        import datetime as _dt
        cur = self.conn.execute(
            """SELECT r.id, p.name, p.commodity, r.started_at, r.status,
                      (SELECT value FROM metrics m
                        WHERE m.run_id = r.id
                          AND m.key = 'cv_auc_mean') AS cv
                 FROM runs r JOIN projects p ON p.id = r.project_id
                ORDER BY r.started_at DESC""")
        out = []
        for rid, name, comm, started, status, cv in cur.fetchall():
            out.append({
                "run": rid, "project": name, "commodity": comm,
                "started": _dt.datetime.fromtimestamp(started)
                              .strftime("%Y-%m-%d %H:%M"),
                "cv_auc": f"{cv:.3f}" if cv is not None else "",
                "status": status})
        return out

    def close(self) -> None:
        self.conn.close()
