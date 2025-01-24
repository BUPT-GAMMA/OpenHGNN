# How to update OpenHGNN documents

## Requirements

- sphinx
- sphinx-gallery 
- sphinx_rtd_theme

Or you can just install with requirements
```
pip install -r ./requirements.txt
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
```
cd docs 
sh build.sh
```
Go to  http://127.0.0.1:8000/ and see whether your edition is displayed correctly.

You can specify the language in conf.py.
### 4. Clean before your submit.
```
cd docs 
sh clean.sh
```