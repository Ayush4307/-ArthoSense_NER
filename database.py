"""
ArthoSense NER - Local SQLite Database Module
Provides offline-first persistent storage, schema migrations, and USB/file sync export.
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "arthosense.db")

def get_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_uid TEXT UNIQUE,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            height_cm REAL,
            weight_kg REAL,
            bmi REAL,
            district TEXT,
            menopause INTEGER DEFAULT 0,
            previous_injury INTEGER DEFAULT 0,
            family_history INTEGER DEFAULT 0,
            activity_level TEXT,
            occupation_loading INTEGER DEFAULT 0,
            pain_score INTEGER DEFAULT 0,
            stiffness_symptom INTEGER DEFAULT 0,
            crepitus_symptom INTEGER DEFAULT 0,
            rom_angle REAL,
            vibration_rms REAL,
            dominant_freq REAL,
            rule_risk_score REAL,
            risk_category TEXT,
            oa_grade TEXT,
            ml_probability REAL,
            clinical_rationale TEXT,
            recommendations TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_screening(data: dict) -> int:
    """Inserts a new patient screening record into SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Generate UID if not provided
    if not data.get("patient_uid"):
        timestamp_str = datetime.now().strftime("%y%m%d%H%M%S")
        data["patient_uid"] = f"NER-OA-{timestamp_str}"
        
    query = """
        INSERT INTO screenings (
            patient_uid, name, age, gender, height_cm, weight_kg, bmi, district,
            menopause, previous_injury, family_history, activity_level,
            occupation_loading, pain_score, stiffness_symptom, crepitus_symptom,
            rom_angle, vibration_rms, dominant_freq, rule_risk_score,
            risk_category, oa_grade, ml_probability, clinical_rationale, recommendations
        ) VALUES (
            :patient_uid, :name, :age, :gender, :height_cm, :weight_kg, :bmi, :district,
            :menopause, :previous_injury, :family_history, :activity_level,
            :occupation_loading, :pain_score, :stiffness_symptom, :crepitus_symptom,
            :rom_angle, :vibration_rms, :dominant_freq, :rule_risk_score,
            :risk_category, :oa_grade, :ml_probability, :clinical_rationale, :recommendations
        )
    """
    cursor.execute(query, data)
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_screenings() -> pd.DataFrame:
    """Fetches all screening records as a Pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM screenings ORDER BY id DESC", conn)
    conn.close()
    return df

def get_screening_by_id(record_id: int) -> dict:
    """Fetches a single screening record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM screenings WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_summary_stats() -> dict:
    """Returns aggregated summary statistics for the local dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM screenings")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM screenings WHERE risk_category LIKE '%High%'")
    high_risk = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM screenings WHERE risk_category LIKE '%Moderate%'")
    mod_risk = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM screenings WHERE risk_category LIKE '%Low%'")
    low_risk = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(age), AVG(bmi), AVG(rom_angle) FROM screenings")
    row = cursor.fetchone()
    avg_age = round(row[0], 1) if row and row[0] else 0.0
    avg_bmi = round(row[1], 1) if row and row[1] else 0.0
    avg_rom = round(row[2], 1) if row and row[2] else 0.0
    
    conn.close()
    return {
        "total_screenings": total,
        "high_risk": high_risk,
        "moderate_risk": mod_risk,
        "low_risk": low_risk,
        "avg_age": avg_age,
        "avg_bmi": avg_bmi,
        "avg_rom": avg_rom
    }

def export_to_csv(output_path: str = "arthosense_screenings_export.csv") -> str:
    """Exports all local records to CSV for field synchronization via USB."""
    df = get_all_screenings()
    df.to_csv(output_path, index=False)
    return output_path

def export_to_json(output_path: str = "arthosense_screenings_export.json") -> str:
    """Exports all local records to JSON format."""
    df = get_all_screenings()
    df.to_json(output_path, orient="records", indent=2)
    return output_path

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully with updated schema.")