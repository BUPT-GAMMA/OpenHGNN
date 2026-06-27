# How to update OpenHGNN documents

## Requirements

- sphinx
- sphinx-rtd-theme

Or you can just install with requirements

```bash
pip install -r docs/requirements.txt
```

## Steps

### 1. Install dependencies.

### 2. Update English Documents.

### 3. Update Chinese translation.

First, update the English text for translation.

```
cd docs
sh update.sh
```
Then edit xx.po files in source/locales/zh_CN/LC_MESSAGES 

### 4. Build and check locally.

The release quality gate uses strict Sphinx mode, so warnings must be fixed
before merging documentation changes:

```bash
python -m sphinx -W -b html docs/source /tmp/openhgnn-docs-build
```

For the legacy local preview flow:

```
cd docs 
sh build.sh
```
Go to  http://127.0.0.1:8000/ and see whether your edition is displayed correctly.

You can specify the language in conf.py.

### 5. Clean before your submit.

```
cd docs 
sh clean.sh
```

## CI and deployment

- `.github/workflows/quality.yml` builds the docs with `sphinx -W` on pull
  requests.
- `.github/workflows/docs.yml` builds the same HTML docs on `main` and deploys
  them with GitHub Pages.
- Sphinx autosummary writes generated files under `docs/source/generated/`;
  this directory is ignored and should not be committed.
