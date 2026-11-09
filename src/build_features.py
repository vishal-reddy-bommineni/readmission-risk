"""Step 1: run the SQL in sql/ against DuckDB and write the modeling table.

Usage:  python -m src.build_features
"""
import duckdb

from src.config import DB_PATH, FEATURES_PARQUET, RAW_CSV, REPORTS_DIR, ROOT


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(
            f"Missing {RAW_CSV}.\nDownload the dataset first - see data/README.md."
        )
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    for sql_file in sorted((ROOT / "sql").glob("*.sql")):
        sql = sql_file.read_text().replace("__RAW_PATH__", RAW_CSV.as_posix())
        con.execute(sql)
        print(f"ran {sql_file.name}")

    audit = con.execute("SELECT * FROM cohort_audit").df()
    audit.to_csv(REPORTS_DIR / "cohort_audit.csv", index=False)
    print("\nCohort audit:\n", audit.to_string(index=False))

    con.execute(f"COPY features TO '{FEATURES_PARQUET.as_posix()}' (FORMAT PARQUET)")
    n, rate = con.execute("SELECT COUNT(*), AVG(label_readmit_30d) FROM features").fetchone()
    print(f"\nWrote {n:,} rows to {FEATURES_PARQUET.name}; 30-day readmission rate = {rate:.3f}")
    con.close()


if __name__ == "__main__":
    main()
