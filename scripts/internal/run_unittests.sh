#!/bin/sh

# Run unittest
#
# This scripts run Python unittest module.

. ./env/bin/activate
python -m unittest discover -s tests_package
