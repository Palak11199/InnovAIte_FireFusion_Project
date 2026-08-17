"""
FireFusion: combine existing ERA5 and ERA5-Land processed exports.

This script DOES NOT download data from Google Earth Engine.
It combines already-exported FireFusion datasets.

Expected storage layout
-----------------------
ERA5:
    One CSV per year, e.g.
    FireFusion_ERA5_Victoria_12Hourly_5kmGrid_2018.csv

ERA5-Land:
    One ZIP per year. Each ZIP may contain multiple CSV chunks.

Both extraction pipelines use:
- 12-hour aggregation
- a nominal 5 km Victoria grid
- datetime / timestamp / interval_start / interval_end metadata
- polygon geometry exported in `.geo`

The merge key is:
    geometry_id + datetime

where geometry_id is a temporary SHA-256 hash derived from `.geo`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


YEARS = range(2018, 2023)
REQUIRED_COLUMNS = {"datetime", ".geo"}
SHARED_METADATA = [".geo", "timestamp", "interval_start", "interval_end"]

def canonical_geometry(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Missing value found in .geo column.")

    try:
        geometry = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON found in .geo column.") from exc

    return json.dumps(geometry, sort_keys=True, separators=(",", ":"))


def create_geometry_id(value: object) -> str:
    canonical = canonical_geometry(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_columns(df: pd.DataFrame, dataset_name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing)}"
        )


def prepare_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    validate_columns(df, dataset_name)

    prepared = df.copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="raise")
    prepared["geometry_id"] = prepared[".geo"].apply(create_geometry_id)

    return prepared


def find_era5_file(era5_dir: Path, year: int) -> Path:
    matches = [
        path
        for path in era5_dir.glob(f"*{year}*.csv")
        if "jan" not in path.name.lower()
        and "era5land" not in path.name.lower()
        and "era5_land" not in path.name.lower()
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one full-year ERA5 CSV for {year} in "
            f"{era5_dir}, but found {len(matches)}: "
            f"{[path.name for path in matches]}"
        )

    return matches[0]


def find_era5_land_zip(era5_land_dir: Path, year: int) -> Path:
    matches = [
        path
        for path in era5_land_dir.glob(f"*{year}*.zip")
        if "test" not in path.name.lower()
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ERA5-Land ZIP for {year} in "
            f"{era5_land_dir}, but found {len(matches)}: "
            f"{[path.name for path in matches]}"
        )

    return matches[0]


def load_era5_year(era5_dir: Path, year: int) -> pd.DataFrame:
    path = find_era5_file(era5_dir, year)
    print(f"  ERA5 file: {path.name}")

    df = pd.read_csv(path, low_memory=False)
    return prepare_dataset(df, f"ERA5 {year}")


def load_era5_land_year(era5_land_dir: Path, year: int) -> pd.DataFrame:
    zip_path = find_era5_land_zip(era5_land_dir, year)
    print(f"  ERA5-Land ZIP: {zip_path.name}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temp_dir = Path(temporary_directory)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir)

        csv_files = sorted(
            path
            for path in temp_dir.rglob("*.csv")
            if not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files were found inside {zip_path.name}."
            )

        print(f"  ERA5-Land chunks found: {len(csv_files)}")

        chunks = []
        for csv_path in csv_files:
            print(f"    - {csv_path.name}")
            chunks.append(pd.read_csv(csv_path, low_memory=False))

        combined_land = pd.concat(chunks, ignore_index=True)

    return prepare_dataset(combined_land, f"ERA5-Land {year}")


def verify_alignment(
    era5: pd.DataFrame,
    era5_land: pd.DataFrame,
    year: int,
) -> None:
    merge_keys = ["geometry_id", "datetime"]

    if era5.duplicated(merge_keys, keep=False).any():
        raise ValueError(
            f"ERA5 {year} contains duplicate geometry-datetime keys."
        )

    if era5_land.duplicated(merge_keys, keep=False).any():
        raise ValueError(
            f"ERA5-Land {year} contains duplicate geometry-datetime keys."
        )

    era5_geometries = set(era5["geometry_id"])
    land_geometries = set(era5_land["geometry_id"])

    if era5_geometries != land_geometries:
        raise ValueError(
            f"{year}: geometry mismatch. "
            f"Only in ERA5: {len(era5_geometries - land_geometries):,}; "
            f"only in ERA5-Land: {len(land_geometries - era5_geometries):,}."
        )

    era5_keys = set(
        map(tuple, era5[merge_keys].itertuples(index=False, name=None))
    )
    land_keys = set(
        map(tuple, era5_land[merge_keys].itertuples(index=False, name=None))
    )

    if era5_keys != land_keys:
        raise ValueError(
            f"{year}: geometry-datetime coverage mismatch. "
            f"Only in ERA5: {len(era5_keys - land_keys):,}; "
            f"only in ERA5-Land: {len(land_keys - era5_keys):,}."
        )

    print(f"  Matching geometries: {len(era5_geometries):,}")
    print(f"  Matching geometry-datetime records: {len(era5_keys):,}")


def verify_shared_metadata(
    era5: pd.DataFrame,
    era5_land: pd.DataFrame,
) -> None:
    merge_keys = ["geometry_id", "datetime"]

    common = [
        column
        for column in SHARED_METADATA
        if column in era5.columns and column in era5_land.columns
    ]

    if not common:
        return

    check = era5[merge_keys + common].merge(
        era5_land[merge_keys + common],
        on=merge_keys,
        how="inner",
        suffixes=("_era5", "_era5land"),
        validate="one_to_one",
    )

    for column in common:
        left = check[f"{column}_era5"]
        right = check[f"{column}_era5land"]
        same = left.eq(right) | (left.isna() & right.isna())

        if not same.all():
            raise ValueError(
                f"Shared metadata '{column}' differs in "
                f"{int((~same).sum()):,} aligned rows."
            )


def combine_year(
    era5: pd.DataFrame,
    era5_land: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    verify_alignment(era5, era5_land, year)
    verify_shared_metadata(era5, era5_land)

    merge_keys = ["geometry_id", "datetime"]

    shared = [
        column
        for column in SHARED_METADATA
        if column in era5.columns and column in era5_land.columns
    ]

    era5_metadata = set(merge_keys + shared)

    era5_for_merge = era5.rename(
        columns={
            column: f"era5_{column}"
            for column in era5.columns
            if column not in era5_metadata
        }
    )

    land_keep = merge_keys + [
        column
        for column in era5_land.columns
        if column not in set(merge_keys + shared)
    ]

    land_for_merge = era5_land[land_keep].rename(
        columns={
            column: f"era5land_{column}"
            for column in land_keep
            if column not in merge_keys
        }
    )

    combined = era5_for_merge.merge(
        land_for_merge,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    first_columns = [
        column
        for column in [
            "geometry_id",
            "datetime",
            ".geo",
            "timestamp",
            "interval_start",
            "interval_end",
        ]
        if column in combined.columns
    ]

    other_columns = [
        column
        for column in combined.columns
        if column not in first_columns
    ]

    return combined[first_columns + other_columns]


def process_year(
    year: int,
    era5_dir: Path,
    era5_land_dir: Path,
    output_dir: Path,
) -> Path:
    print(f"\n===== Processing {year} =====")

    era5 = load_era5_year(era5_dir, year)
    era5_land = load_era5_land_year(era5_land_dir, year)

    print(f"  ERA5 rows loaded: {len(era5):,}")
    print(f"  ERA5-Land rows loaded: {len(era5_land):,}")

    combined = combine_year(era5, era5_land, year)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"FireFusion_ERA5_ERA5Land_Victoria_12Hourly_5kmGrid_{year}.csv"
    )

    combined.to_csv(output_path, index=False)

    print(f"  Combined rows: {len(combined):,}")
    print(f"  Combined columns: {len(combined.columns):,}")
    print(f"  Saved: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine existing FireFusion ERA5 yearly CSVs with "
            "ERA5-Land yearly ZIP archives."
        )
    )

    parser.add_argument(
        "--era5-dir",
        required=True,
        type=Path,
        help="Folder containing full-year ERA5 CSV files.",
    )
    parser.add_argument(
        "--era5-land-dir",
        required=True,
        type=Path,
        help="Folder containing yearly ERA5-Land ZIP files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder for combined yearly CSV outputs.",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=list(YEARS),
        help="Optional: process only one year (2018-2022).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    years = [args.year] if args.year else list(YEARS)

    outputs = []
    for year in years:
        outputs.append(
            process_year(
                year=year,
                era5_dir=args.era5_dir,
                era5_land_dir=args.era5_land_dir,
                output_dir=args.output_dir,
            )
        )

    print("\n===== Complete =====")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()