# Epilogue analytics

A Streamlit dashboard for the Epilogue capstone project: the data, the two regression models and what drives their predictions, the evaluation of the book summaries written by a language model, and the services the app uses.

## Run

```bash
pip install -r analytics/requirements.txt
streamlit run analytics/app.py
```

Run both commands from the repository root. Streamlit reads the Folio theme from `.streamlit/config.toml` there.

## Data

`analytics/data` holds a copy of the processed files the pages read. `analytics/data/MANIFEST.json` lists each file with its size and SHA-256. The two out-of-fold prediction files keep only the three columns the dashboard uses. To rebuild the copy from a full data folder:

```bash
python analytics/bundle_data.py --source <data folder> --out analytics/data
```

The data, model and feature figures come from the Open Library data dumps, and Open Library asks that contributions be given under the CC0 1.0 Universal licence. The summary evaluation figures come from the project's own evaluation runs, and the reader personas are synthetic.

Set `CAPSTONE_DATA` to read from a different data folder.

## Tests

```bash
python -m pytest tests
```
