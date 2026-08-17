# ERA5 and ERA5-Land Dataset Combination

## Overview
This folder contains the script used to combine the existing FireFusion ERA5 and ERA5-Land datasets into a single dataset for each year.

Both datasets have already been collected and processed separately. Therefore, this script does not download or extract new data from Google Earth Engine. 

Instead it: 
1. Loads the existing ERA5 dataset
2. Loads the existing ERA5-Land dataset
3. Handles the different storage formats used by the two datasets
4. Matches records spatially and temporally
5. Combines the ERA5 and ERA5-Land variables.
6. Saves the result as a new CSV file.

The current datasets covering Victoria from 2018 to 2022 are in the shared drive.

## Files
 `combine-era5-era5land.py` : Main Python script for combining ERA5 and ERA5-Land. 

 `README.md` : Instructions and important information about running and understanding the combination process. 

## Dataset Locations
 The source datasets are stored in the FireFusion SharePoint under:
#### ERA5
`AI Modelling/ Processed GEE Training Dataset (CSV) / ERA5 Dataset`

ERA5 is currently stored as one CSV file per year. 

Example:
```text
FireFusion_ERA5_Victoria_12Hourly_5kmGrid_2018.csv
```

#### ERA5
`AI Modelling/ Processed GEE Training Dataset (CSV) / ERA5-Land Dataset`

ERA5-Land is currently stored as ZIP files by year. 

Example:
```text
ERA5_Land_2018.zip
```

Each yearly ZIP contains the ERA5-Land CSV chunks for that year, such as:
```text
FireFusion_ERA5_Land_Victoria_2018_Jan_Jun_12Hourly_5kmGrid.csv
FireFusion_ERA5_Land_Victoria_2018_Jul_Dec_12Hourly_5kmGrid.csv
```
The script handles these chunks and combines them for processing.

## Before Running the Script
The ERA5 and ERA5-Land datasets are not stored inside the GitHub repository because the files are very large. Team members must first download the required source datasets from the FireFusion SharePoint. 

For example, to process 2018, prepare the following structure:
```text
Downloads/
|
|- FireFusion_Test/
      |
      |-- combine-era5-era5land.py
      |-- ERA5 Dataset/
      |   `-- FireFusion_ERA5_Victoria_12Hourly_5kmGrid_2018.csv
      |-- ERA5-Land Dataset/
      |   `-- ERA5_Land_2018.zip
      `-- Combined Dataset/
```

The 'Combined Dataset" folder is used for the generated output. The folder names do not have to be exactly the same as the example above because their locations can be supplied when running the script.

## Requirements
The script requires:
- Python 3
- pandas

Install pandas if it is not already available: 
```bash
pip install pandas
```

You can check the installed pandas version using:
```bash
python3 -c"import pandas; print(pandas.__version__)"
```

## Running the Script
Open Terminal and navigate to the directory containing the script and dataset folders. 

For example:
```bash
cd ~/Downloads/FireFusion_Test
```

To combine the 2018 datasets: 
```bash
python3 combine-era5-era5land.py \
  --era5-dir "ERA5 Dataset" \
  --era5-land-dir "ERA5-Land Dataset" \
  --output-dir "Combined Dataset" \
  --year 2018
```

The arguments mean:
|Argument|Description|
|--|--|
|`--era5-dir`|Folder containing the ERA5 yearly CSV files|
|`--era5-land-dir`|Folder containing the ERA5-Land yearly ZIP files|
|`--output-dir`|Folder where combined CSV will be saved|
|`--year`|Year to process|

## Processing Another Year
Change the `--year` value to process another available year. 

For example, for 2019:
```bash
python3 combine-era5-era5land.py \
  --era5-dir "ERA5 Dataset" \
  --era5-land-dir "ERA5-Land Dataset" \
  --output-dir "Combined Dataset" \
  --year 2019
```

The same approach can be used for the available datasets from 2018 to 2022, provided the corresponding ERA5 and ERA5-Land source files have been downloaded.

## Using Different Folder Locations
The datasets do not need to be stored beside the script. Different paths can be supplied through the command-line arguments.

For example: 
```bash
python3 combine-era5-era5land.py \
  --era5-dir "/Users/example/FireFusion/ERA5 Dataset" \
  --era5-land-dir "/Users/example/FireFusion/ERA5-Land Dataset" \
  --output-dir "/Users/example/FireFusion/Combined Dataset" \
  --year 2018
```

This allows team members to reuse the script without changing the Python source code to match their own computer.

## How the Datasets Are Matched
ERA5 and ERA5-Land contain environmental measurements associated with the FireFusion spatial grid.

Previous investigation of the datasets found that the numeric `grid_id` should **not be relied on by itself as the spatial matching identifier**.

The grid cell geometry stored in `.geo` represents the actual spatial cell.

Therefore, the combination process uses the grid geometry to establish spatial correspondence between ERA5 and ERA5-Land.

Temporal information is also required because each grid cell contains observations across multiple timestamps. 

Conceptually, records are matched using:
```text
grid cell geometry + datetime
```

rather than:
```text
grid_id only
```

This ensures that a record represents the same spatial location and observation time before ERA5 and ERA5-Land variables are combined. For the detailed investigation of ERA5 and ERA5-Land grid alignment, refer to the grid-alignment verification notebook and its documentation in the research notebooks folder.

## Spatial Resolution
The source ERA5 and ERA5-Land datasets were prepared using the FireFusion 5 km grid. The combination script does **not** generate a new spatial grid. I uses the spatial information already contained in the processed source datasets. 

Therefore, the expected combined output retains the existing:
```text
5 km grid
```

## Temporal Resolution
The intended FireFusion datasets use a 12-hour temporal resolution. 

The expected observation times are: 
```text
00:00
12:00
```

For a normal 365 day year, this means the expected number of timestamps is: 
```text
365 days x 2 observations per day = 730 timestamps
```

During testing of the 2018 combined output, the following results were observed:
```text
Earliest datetime: 2018-01-01 00:00:00
Latest datetime: 2018-12-31 12:00:00
Unique datetimes: 730
Unique hours: [0,12]
```

This confirms that the tested 2018 output retained the expected 12-hour temporal structure.

## Output
The script generates a combined CSV for the selected year. 

For example:
```text
Combined Dataset/
|--FireFusion_ERA5_ERA5Land_Victoria_12Hourly_5kmGrid_2018.csv
```

The resulting dataset contains variables from both ERA5 and ERA5-Land. Feature columns are identified so that their source dataset remains clear.

For example:
```text
era5_
era5land_
```

The shared spatial and temporal information is retained so the resulting dataset can be used for later FireFusion preprocessing, analysis, feature selection, and modelling.

## Expected Output Size
The combined datasets are very large. For example, the 2018 combined CSV generated during testing was several gigabytes in size. 

Because:
- Excel may not be able to open the complete file.
- A failure to open the file in a spreadsheet application does not necessarily mean that the output is corrupted.
- Python/pandas should be used to inspect and validate the output.
- Make sure sufficient disk space is available before processing multiple years.
- Loading the entire combined dataset into memory may require significant RAM.

## Checking the Output
Because the combined CSV can be several gigabytes, it is recommended to inspect a small portion of the file instead of loading the entire dataset. 

For example: 
```python
import pandas as pd

file_path = (
    "Combined Dataset/"
    "FireFusion_ERA5_ERA5Land_Victoria_12Hourly_5kmGrid_2018.csv"
)

sample = pd.read_csv(file_path, nrows=10)

print(sample)
print(sample.columns.tolist())
```

This allows the output structure and columns to be checked without loading the entire dataset into memory.

## Checking the Datetime Coverage
The datetime column can be loaded separately to reduce memory usage.
```python
import pandas as pd

file_path = (
    "Combined Dataset/"
    "FireFusion_ERA5_ERA5Land_Victoria_12Hourly_5kmGrid_2018.csv"
)

date_check = pd.read_csv(
    file_path,
    usecols=["datetime"]
)

date_check["datetime"] = pd.to_datetime(
    date_check["datetime"],
    format="mixed",
    errors="raise"
)

print("Earliest datetime:", date_check["datetime"].min())
print("Latest datetime:", date_check["datetime"].max())
print("Unique datetimes:", date_check["datetime"].nunique())
```

For the tested 2018 output, the expected result is: 
```text
Earliest datetime: 2018-01-01 00:00:00
Latest datetime: 2018-12-31 12:00:00
Unique datetimes: 730
```

## Checking the 12-Hour Intervals
After parsing the datetime column:
```python
date_check["hour"] = date_check["datetime"].dt.hour

print("Unique hours:")
print(sorted(date_check["hour"].unique()))

print("\nNumber of rows by hour:")
print(date_check["hour"].value_counts().sort_index())
```

For the tested 2018 output, the expected unique hours are: 
```text
[0, 12]
```

This indicates that the combined dataset contains the intended midnight and midday observations.

## Important Datetime Parsing Note
When validating the output, pandas must correctly parse the complete datetime values. 

Use: 
```python
date_check["datetime"] = pd.to_datetime(
    date_check["datetime"],
    format="mixed",
    errors="raise"
)
```

During testing, parsing the datetime incorrectly could make the data appear to contain only daily observations. 

For example: 
```text
2018-01-01
2018-01-02
2018-01-03
...
```

This can incorrectly suggest that there are only:
```text
365 unique dates
```

After correctly parsing the complete timestamps, the 2018 output contains:
```text
730 unique datetimes
```

with:
```text
00:00
12:00
```

Therefore, datetime parsing should be checked before concluding that the 12-hour observations are missing. 

## Validation Performed
The combination workflow was initially tested using the 2018 ERA5 and ERA5-Land datasets. 

The generated output was checked for:
- successful creation of the combined CSV
- presence of ERA5 variables
- presence of ERA5-Land variables
- full-year datetime coverage
- preservation of the expected 12-hour temporal intervals
- expected spatial matching structure

The tested 2018 output returned:
```text
Earliest datetime: 2018-01-01 00:00:00
Latest datetime:   2018-12-31 12:00:00
Unique datetimes:  730
Unique hours:      [0, 12]
```

This is consistent with: 
```text
365 days × 2 observations per day = 730 timestamps
```

The remaining years should be validated after they are processed rather than assuming that all years produce identical results. 

## Troubleshooting
### File Not Found
Check that:
- the required ERA5 yearly CSV has been downloaded
- the required ERA5-Land yearly ZIP has been downloaded
- the requested year matches the downloaded files
- the directory paths supplied to the script are correct

### ERA5-Land ZIP Cannot Be Read
Check the contents of the ZIP archive. The archive contains the ERA5-Land CSV chunks required for the selected year. If duplicate, hidden, or unrelated files are present, inspect the archive before running the script again. 

### Script Takes a Long Time
This is expected. The source datasets contain millions of records, and the ERA5 yearly CSV files can be several gigabytes in size. 

The script needs to:
1. Read the source files
2. Process the ERA5-Land chunks
3. Prepare the matching fields
4. Match the datasets
5. Combine the columns
6. Write a new multi-gigabyte CSV

Processing time will depend on:
- dataset size
- available RAM
- CPU performance
- storage speed

Do not assume the script has failed simply because it takes several minutes.

### Output Cannot Be Opened in Excel or Numbers
This is expected for very large CSV files. Use Python/pandas to inspect the output instead. 

For example: 
```python
sample = pd.read_csv(
    "Combined Dataset/FireFusion_ERA5_ERA5Land_Victoria_12Hourly_5kmGrid_2018.csv",
    nrows=20
)

print(sample)
```

### Only 365 Dates Appear During Validation
Do not immediately conclude that half of the observations are missing. Check that the complete datetime values have been parsed correctly. 

Use:
```python
date_check["datetime"] = pd.to_datetime(
    date_check["datetime"],
    format="mixed",
    errors="raise"
)
```

Then check: 
```python
print(date_check["datetime"].nunique())
print(sorted(date_check["datetime"].dt.hour.unique()))
```

For the tested 2018 output, this should return:
```text
730
[0, 12]
```

## Workflow Summary
The complete workflow is:
1. Download existing ERA5 dataset
2. Download existing ERA5-Land dataset
3. Prepare local folders
4. Run combine-era5-era5land.py
5. Load ERA5 yearly CSV
6. Load ERA5-Land yearly ZIP/chunks
7. Prepare spatial and temporal matching
8. Match geometry and datetime
9. Combine ERA5 and ERA5-Land variables
10. Save combined yearly CSV
11. Validate output

## Purpose in FireFusion
This script provides a reusable preprocessing step for the FireFusion AI Modelling stream. Instead of recollecting or manually combining ERA5 and ERA5-Land data, team members can use the existing processed datasets and run the same combination workflow for each year. 

The resulting dataset provides ERA5 and ERA5-Land environmental variables in a single spatially and temporally aligned dataset that can be used in later FireFusion data preparation and modelling tasks. 
