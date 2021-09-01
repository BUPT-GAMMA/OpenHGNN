How to build document locally
================================

Requirements
------------
* sphinx
* sphinx-gallery
* sphinx_rtd_theme

Or you can just install with requirements
```
pip install -r ./requirements.txt
```

Build documents
---------------
First, clean up existing files:
```bash
./clean.sh
```

Then build:
```bash
make html
```

Render locally
--------------
```bash
cd build/html
python -m http.server 8000
```
