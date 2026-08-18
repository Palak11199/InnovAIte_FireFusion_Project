"""
FireFusion Data Pipeline — ERA5 / ERA5-Land Weather Reanalysis
=============================================================
Pipeline: C. Weather Ingestion (real ERA5 data, replacing the unrecoverable
          21,888 legacy weather_observation rows going forward)

Source: Copernicus Climate Data Store (CDS)
    https://cds.climate.copernicus.eu/

See CDS_Setup.md for the one-time account/license/API-key setup this
script depends on — that part is manual and can't be automated.

WHY TWO DATASETS:
    The seven target fields split cleanly by naming prefix, which almost
    certainly reflects which underlying CDS dataset each was meant to come
    from:
        era5land_*  -> "reanalysis-era5-land"           (~9km resolution)
            - era5land_temperature_2m_c
            - era5land_surface_solar_radiation_downwards
            - era5land_skin_temperature_c
        era5_*      -> "reanalysis-era5-single-levels"   (~31km resolution)
            - era5_dewpoint_temperature_2m_c
            - era5_total_precipitation
            - era5_u_component_of_wind_10m
            - era5_v_component_of_wind_10m
    This script queries both and merges the results per location/hour.

IMPORTANT — THINGS TO VERIFY BEFORE TRUSTING OUTPUT VALUES:
    1. Solar radiation is an ACCUMULATED field in ERA5/ERA5-Land (joules
       per square metre, accumulated from the start of the forecast
       period), not an instantaneous flux. Depending on which hours you
       request, you may need to de-accumulate (subtract the previous
       hour's cumulative value) to get a true hourly value. This script
       requests single hours and does NOT attempt de-accumulation —
       verify against a known reference value before trusting it, or ask
       whoever defined the AI model's expected units/convention for this
       field.
    2. total_precipitation is left in ERA5's native units (metres,
       accumulated over the hour). Convert to mm (x1000) if the AI model
       expects mm — check before assuming.
    3. Wind u/v components and temperature/dewpoint/skin-temperature unit
       conversions (Kelvin -> Celsius for temperature fields) ARE handled
       below, but double-check a few output rows against an independent
       source (e.g. the CDS web UI's own preview) before trusting a full
       backfill.
    4. CDS API request syntax has changed before during Copernicus
       infrastructure migrations. If retrieve() fails with a parameter
       error, check the dataset's "API request" tab on the CDS website
       for the current exact parameter names before assuming this script
       is wrong in a deeper way.

LATENCY:
    Unlike FIRMS (near-real-time), ERA5/ERA5-Land is not available for
    "today" — there is a publication delay. This script requests data for
    (today - LAG_DAYS), not today; LAG_DAYS defaults to 6 as a safe
    buffer. If a request comes back empty, try increasing it.

Developer Instructions compliance (matching extract_firms.py's pattern):
    1. Source documented above (URLs + comments).
    2. Reuses the EXISTING Location_Registry rows (the 4 known grid
       cells) rather than creating new ones — this script does not
       snap/create new locations.
    3. Raw values preserved as returned by CDS; unit conversions are
       explicit and commented, not silent.
    4. Data typing enforced before insert.
    5. NO hardcoded credential fallbacks — unlike extract_firms.py's
       current defaults (flagged separately as a fix-me), every secret
       here is required from the environment and the script fails loudly
       if one is missing, rather than silently using a real value baked
       into the code.
"""

import os
import sys
import math
import glob
import zipfile
import logging
from datetime import datetime, timedelta, timezone

import cdsapi
import xarray as xr
import pandas as pd
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------
# Config — all required, no fallback defaults (see docstring point 5)
# ---------------------------------------------------------------------
def require_env(name):
    val = os.environ.get(name)
    if not val:
        print(f"FATAL: required environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val

DB_CONFIG = dict(
    host=require_env("SUPABASE_HOST"),
    dbname=require_env("SUPABASE_DB"),
    user=require_env("SUPABASE_USER"),
    password=require_env("SUPABASE_PASSWORD"),
    port=int(os.environ.get("SUPABASE_PORT", "5432")),
    sslmode="require",
)

# cdsapi reads its key from ~/.cdsapirc by default. If CDS_API_KEY /
# CDS_API_URL are provided as env vars instead (e.g. in CI), write them
# out to that file at runtime so the client picks them up.
CDS_API_KEY = require_env("CDS_API_KEY")
CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")

cdsapirc_path = os.path.expanduser("~/.cdsapirc")
if not os.path.exists(cdsapirc_path):
    with open(cdsapirc_path, "w") as f:
        f.write(f"url: {CDS_API_URL}\nkey: {CDS_API_KEY}\n")

LAG_DAYS = int(os.environ.get("ERA5_LAG_DAYS", "6"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("era5_pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("era5_pipeline")

ERA5LAND_VARS = ["2m_temperature", "skin_temperature", "surface_solar_radiation_downwards"]
ERA5_VARS = ["2m_dewpoint_temperature", "total_precipitation",
             "10m_u_component_of_wind", "10m_v_component_of_wind"]


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------
def _load_dataset(path):
    """Load a CDS download that may be a single NetCDF file, a ZIP
    archive of multiple per-variable NetCDF files (confirmed behavior as
    of run_id 17 — CDS's current API bundles this multi-variable
    data_format=netcdf request into a zip rather than one combined
    file), or an error response saved where real data was expected
    (license not accepted / bad API key). Handles all three explicitly
    instead of letting xarray fail with its generic "no matching IO
    backend" error."""
    with open(path, "rb") as f:
        head = f.read(500)

    if head[:2] == b"PK":
        extract_dir = path + "_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        nc_files = sorted(glob.glob(os.path.join(extract_dir, "*.nc")))
        if not nc_files:
            raise RuntimeError(
                f"{path} is a ZIP archive but contains no .nc files. "
                f"Contents: {os.listdir(extract_dir)}"
            )
        log.info(f"{path}: ZIP archive with {len(nc_files)} NetCDF file(s) — merging: {nc_files}")
        # combine="by_coords" merges the per-variable files into one
        # Dataset as long as they share the same lat/lon/time grid,
        # which they should since they came from the same request.
        return xr.open_mfdataset(nc_files, combine="by_coords")

    stripped = head.strip()
    if stripped.startswith(b"{") or stripped.startswith(b"<"):
        raise RuntimeError(
            f"{path} looks like an error response (JSON/HTML), not real "
            f"NetCDF data — the request was likely rejected server-side "
            f"even though retrieve() didn't raise. First bytes:\n"
            f"{head!r}\n"
            f"Most likely causes: (1) the license for this dataset "
            f"hasn't been accepted yet on the CDS website (see "
            f"CDS_Setup.md step 2 — each dataset's license is separate "
            f"and both must be accepted), or (2) CDS_API_KEY is missing, "
            f"malformed, or a placeholder rather than your real key."
        )

    log.info(f"{path}: looks like a single binary NetCDF file — opening directly.")
    return xr.open_dataset(path)


def fetch_era5_land(target_date, lat, lon, out_path):
    """Pull the 3 era5land_* fields for one grid cell, one date, all
    24 hours. Returns the xarray Dataset."""
    client = cdsapi.Client()
    # Small bounding box centred on the point so the nearest ERA5-Land
    # grid cell to (lat, lon) is included in the response.
    area = [lat + 0.05, lon - 0.05, lat - 0.05, lon + 0.05]  # N, W, S, E
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ERA5LAND_VARS,
            "year": f"{target_date.year:04d}",
            "month": f"{target_date.month:02d}",
            "day": f"{target_date.day:02d}",
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "data_format": "netcdf",
        },
        out_path,
    )
    return _load_dataset(out_path)


def fetch_era5_single_levels(target_date, lat, lon, out_path):
    """Pull the 4 era5_* fields for one grid cell, one date, all 24 hours."""
    client = cdsapi.Client()
    area = [lat + 0.15, lon - 0.15, lat - 0.15, lon + 0.15]  # coarser grid, wider box
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_VARS,
            "year": f"{target_date.year:04d}",
            "month": f"{target_date.month:02d}",
            "day": f"{target_date.day:02d}",
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "data_format": "netcdf",
        },
        out_path,
    )
    return _load_dataset(out_path)


# ---------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------
def _clean_float(v):
    """CDS/ERA5-Land can return NaN for a grid point outside that
    dataset's actual coverage — confirmed on a real run: two of the four
    weather grid cells sit right at Victoria's coastal boundary, and
    reanalysis-era5-land (land-only) returned NaN for them while
    reanalysis-era5-single-levels (covers ocean too) returned real
    values for the same points. NaN is a valid IEEE-754 float and
    inserts into a double precision column silently — a plain
    `IS NOT NULL` check downstream will NOT catch it, which makes an
    un-cleaned NaN worse than an honest NULL, not better. Convert NaN to
    real Python None (SQL NULL) so missing coverage is represented
    honestly instead of as a poisoned numeric value."""
    if v is None:
        return None
    fv = float(v)
    return None if math.isnan(fv) else fv


def kelvin_to_celsius(k):
    k = _clean_float(k)
    return None if k is None else k - 273.15


def _time_dim_name(ds):
    """CDS's returned datasets have been observed using 'valid_time' as
    the time coordinate rather than 'time' (confirmed on a real run —
    every reanalysis-era5-land response so far has come back with
    dimensions {'valid_time': 24}, not 'time'). Detect whichever name is
    actually present instead of hardcoding one, since this has already
    proven inconsistent across CDS API versions/datasets."""
    for candidate in ("time", "valid_time"):
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise RuntimeError(
        f"Neither 'time' nor 'valid_time' found in dataset coordinates: "
        f"{list(ds.coords)}. CDS may have changed naming again — check "
        f"the actual coordinate names for this response and add the new "
        f"one to the candidates list above."
    )


def _select_nearest_time(ds, time_dim, target_ts):
    """Select the timestep nearest to target_ts by position (isel), not
    by value (sel) — this sidesteps dtype-mismatch errors between
    xarray's internal datetime64 handling and Python's tz-aware datetime
    objects. Confirmed on a real run: one dataset came back as
    datetime64[us, UTC] (tz-aware, microsecond precision) while our query
    value got coerced to datetime64[ns] (naive, nanosecond precision),
    and xarray's .sel(method='nearest') refuses to compare the two
    dtypes. Normalizing everything to plain tz-naive UTC pandas
    Timestamps before finding the nearest index avoids that entirely,
    regardless of whatever dtype/precision CDS returns next time."""
    coord_times = pd.to_datetime(ds[time_dim].values, utc=True).tz_localize(None)
    target_naive = pd.Timestamp(target_ts).tz_localize(None) if pd.Timestamp(target_ts).tzinfo else pd.Timestamp(target_ts)
    idx = int(abs(coord_times - target_naive).argmin())
    return ds.isel(**{time_dim: idx})


def extract_hourly_rows(ds_land, ds_single, lat, lon, target_date):
    """Select the nearest grid point from each dataset and return one
    row per hour with all 7 ERA5 fields merged."""
    rows = []
    land_time_dim = _time_dim_name(ds_land)
    single_time_dim = _time_dim_name(ds_single)
    log.info(f"Time dimension detected: land='{land_time_dim}', single-levels='{single_time_dim}'")

    for hour in range(24):
        ts = datetime(target_date.year, target_date.month, target_date.day,
                       hour, 0, tzinfo=timezone.utc)

        land_pt = ds_land.sel(latitude=lat, longitude=lon, method="nearest")
        single_pt = ds_single.sel(latitude=lat, longitude=lon, method="nearest")

        try:
            land_hour = _select_nearest_time(land_pt, land_time_dim, ts)
            single_hour = _select_nearest_time(single_pt, single_time_dim, ts)
        except Exception as e:
            log.error(f"Time selection failed for hour {hour}: {e}")
            continue

        row = {
            "datetime_record": ts,
            "era5land_temperature_2m_c": kelvin_to_celsius(land_hour["t2m"].values),
            "era5land_skin_temperature_c": kelvin_to_celsius(land_hour["skt"].values),
            # Left in native accumulated J/m^2 — see de-accumulation caveat above.
            "era5land_surface_solar_radiation_downwards": _clean_float(land_hour["ssrd"].values),
            "era5_dewpoint_temperature_2m_c": kelvin_to_celsius(single_hour["d2m"].values),
            # Left in native metres — convert x1000 for mm if required.
            "era5_total_precipitation": _clean_float(single_hour["tp"].values),
            "era5_u_component_of_wind_10m": _clean_float(single_hour["u10"].values),
            "era5_v_component_of_wind_10m": _clean_float(single_hour["v10"].values),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
def get_existing_locations(cur):
    """Target only grid cells that actually have weather data, not every
    row in Location_Registry. Location_Registry is a shared hub also
    populated by the FIRMS fire pipeline (confirmed: it had grown to 99
    rows, only a handful of which are weather-relevant) — pulling all of
    it here would submit CDS requests for irrelevant fire-hotspot cells
    and blow well past the workflow's timeout. Deriving the location set
    from weather_observation itself keeps this correct automatically:
    after the location_id repair, this returns the real historical
    weather locations, and any location this script itself inserts data
    for going forward is legitimately weather-relevant by definition."""
    cur.execute(
        """
        SELECT DISTINCT w.location_id, l.grid_latitude, l.grid_longitude
        FROM weather_observation w
        JOIN location_registry l ON l.location_id = w.location_id
        ORDER BY w.location_id;
        """
    )
    return cur.fetchall()


def get_or_create_time(cur, dt):
    cur.execute(
        """
        INSERT INTO Time_Registry (datetime_record, season)
        VALUES (%s, %s)
        ON CONFLICT (datetime_record) DO UPDATE SET season = EXCLUDED.season
        RETURNING time_id
        """,
        (dt, None),
    )
    return cur.fetchone()[0]


def start_run_log(cur, pipeline_name, sources_attempted):
    cur.execute(
        """
        INSERT INTO Pipeline_Run_Log (pipeline_name, sources_attempted, status)
        VALUES (%s, %s, 'RUNNING')
        RETURNING run_id
        """,
        (pipeline_name, ",".join(sources_attempted)),
    )
    return cur.fetchone()[0]


def finish_run_log(cur, run_id, status, sources_succeeded, rows_fetched, rows_inserted, rows_skipped, error_message=None):
    cur.execute(
        """
        UPDATE Pipeline_Run_Log
        SET finished_at = now(), status = %s, sources_succeeded = %s,
            rows_fetched = %s, rows_inserted = %s, rows_skipped = %s, error_message = %s
        WHERE run_id = %s
        """,
        (status, ",".join(sources_succeeded), rows_fetched, rows_inserted, rows_skipped, error_message, run_id),
    )


def insert_weather_row(cur, location_id, time_id, lat, lon, fields):
    # NOTE: weather_observation currently has no UNIQUE constraint on
    # (location_id, time_id), unlike Fire_Incident_Record's uq_firms_pixel.
    # Recommended before running this in production:
    #   ALTER TABLE weather_observation
    #     ADD CONSTRAINT uq_weather_location_time UNIQUE (location_id, time_id);
    # Without it, re-running this script for the same hour will insert
    # duplicate rows rather than updating in place.
    cur.execute(
        """
        INSERT INTO weather_observation (
            location_id, time_id, original_latitude, original_longitude,
            era5land_temperature_2m_c, era5_dewpoint_temperature_2m_c,
            era5_total_precipitation, era5_u_component_of_wind_10m,
            era5_v_component_of_wind_10m, era5land_surface_solar_radiation_downwards,
            era5land_skin_temperature_c
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            location_id, time_id, lat, lon,
            fields["era5land_temperature_2m_c"], fields["era5_dewpoint_temperature_2m_c"],
            fields["era5_total_precipitation"], fields["era5_u_component_of_wind_10m"],
            fields["era5_v_component_of_wind_10m"], fields["era5land_surface_solar_radiation_downwards"],
            fields["era5land_skin_temperature_c"],
        ),
    )
    return cur.rowcount


def run():
    target_date = (datetime.now(timezone.utc) - timedelta(days=LAG_DAYS)).date()
    log.info(f"ERA5 ingest starting for target_date={target_date} (LAG_DAYS={LAG_DAYS})")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    sources_attempted = ["reanalysis-era5-land", "reanalysis-era5-single-levels"]
    run_id = start_run_log(cur, "era5_weather", sources_attempted)
    conn.commit()

    locations = get_existing_locations(cur)
    log.info(f"Ingesting for {len(locations)} existing grid cells: {locations}")

    total_fetched, total_inserted, total_skipped = 0, 0, 0
    sources_succeeded = []
    error_message = None

    try:
        for location_id, lat, lon in locations:
            land_path = f"/tmp/era5land_{location_id}.nc"
            single_path = f"/tmp/era5single_{location_id}.nc"

            ds_land = fetch_era5_land(target_date, lat, lon, land_path)
            if "reanalysis-era5-land" not in sources_succeeded:
                sources_succeeded.append("reanalysis-era5-land")

            ds_single = fetch_era5_single_levels(target_date, lat, lon, single_path)
            if "reanalysis-era5-single-levels" not in sources_succeeded:
                sources_succeeded.append("reanalysis-era5-single-levels")

            rows = extract_hourly_rows(ds_land, ds_single, lat, lon, target_date)
            total_fetched += len(rows)

            for row in rows:
                time_id = get_or_create_time(cur, row["datetime_record"])
                fields = {k: v for k, v in row.items() if k != "datetime_record"}
                try:
                    rc = insert_weather_row(cur, location_id, time_id, lat, lon, fields)
                    if rc:
                        total_inserted += 1
                    else:
                        total_skipped += 1
                    conn.commit()
                except Exception as e:
                    log.error(f"Insert failed for location_id={location_id} hour={row['datetime_record']}: {e}")
                    conn.rollback()
                    total_skipped += 1

        status = "SUCCESS" if len(sources_succeeded) == 2 else "PARTIAL"

    except Exception as e:
        error_message = str(e)
        status = "FAILED"
        log.error(f"ERA5 pipeline run failed: {e}")

    finish_run_log(cur, run_id, status, sources_succeeded, total_fetched, total_inserted, total_skipped, error_message)
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Run complete. Status={status} Fetched={total_fetched} Inserted={total_inserted} Skipped={total_skipped}")


if __name__ == "__main__":
    run()
