#!/bin/sh

sh clean.sh
make html
cd build/html
python -m http.server 8000

