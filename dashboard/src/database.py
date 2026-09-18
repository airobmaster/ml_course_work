"""SQLite connection + query execution helpers."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "earthquake.db"
TABLE = "buildings"


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run "
            "`python dashboard/scripts/build_db.py` first."
        )
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def run_query(sql: str, params: tuple | list = ()) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query(sql, conn, params=params)
