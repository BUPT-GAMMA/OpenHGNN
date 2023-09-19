#!/bin/sh
# update text for translation

cd source
sphinx-build -b gettext . _build/gettext
sphinx-intl update -p _build/gettext -l zh_CN
