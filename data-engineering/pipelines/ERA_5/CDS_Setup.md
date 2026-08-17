# FireFusion — One-Time Copernicus CDS Setup for ERA5 Ingestion

This is the account/credential setup `era5_ingest.py` needs. It's a one-time,
manual step — creating accounts and accepting license terms isn't something
that can be automated on your behalf, so this is written as instructions
for you to follow directly, same spirit as `GitHub_Actions_Setup.md` but
for the Copernicus Climate Data Store instead of GitHub Secrets.

## 1. Register a free CDS account

Go to https://cds.climate.copernicus.eu/ and create an account (or sign in
via an existing ECMWF account if you have one). Free, no payment info required.

## 2. Accept the dataset license terms

This pipeline pulls from **two separate CDS datasets**, matching the field
naming convention already used in this project (`era5land_*` vs `era5_*`):

| Dataset | Fields it provides here |
|---|---|
| `reanalysis-era5-land` | `era5land_temperature_2m_c`, `era5land_surface_solar_radiation_downwards`, `era5land_skin_temperature_c` |
| `reanalysis-era5-single-levels` | `era5_dewpoint_temperature_2m_c`, `era5_total_precipitation`, `era5_u_component_of_wind_10m`, `era5_v_component_of_wind_10m` |

Each dataset has its own license you must accept **once**, on its product
page, before API requests against it will succeed:
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land — click "Download", scroll to "Terms of use", accept.
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels — same process.

If you skip this, `era5_ingest.py` will fail with a license/permission
error the first time it hits whichever dataset you didn't accept.

## 3. Get your API key

Go to https://cds.climate.copernicus.eu/how-to-api — while logged in, this
page shows your personal API key and the exact `~/.cdsapirc` format
expected by the `cdsapi` Python client. **The exact format has changed
before as CDS has migrated infrastructure, so use whatever this page shows
you at the time you set this up rather than trusting a hardcoded example.**

## 4. Store the key as a secret, never in code

Add it to GitHub the same way you already did for `FIRMS_MAP_KEY` and the
Supabase credentials:

**Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|---|---|
| `CDS_API_KEY` | your personal CDS API key from step 3 |

`era5_ingest.py` reads this from an environment variable with **no
hardcoded fallback** — unlike `extract_firms.py`'s current default values
(flagged separately as something to fix), this script will simply fail
loudly if the secret isn't set, rather than silently falling back to a
real credential embedded in the code.

## 5. Test it manually before trusting the schedule

Run it once via **Actions tab → "FireFusion - ERA5 Weather Pipeline" →
Run workflow**, same as you already do for FIRMS, before relying on the
automatic schedule. ERA5 requests are queued server-side by Copernicus and
can take anywhere from under a minute to over an hour depending on load —
don't be surprised if the first run takes a while, and check the workflow
log rather than assuming a slow run means it's broken.
