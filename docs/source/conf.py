# ==============================================================================
#  C O P Y R I G H T
# ------------------------------------------------------------------------------
#  Copyright (c) 2024 by Robert Bosch GmbH. All rights reserved.
#
#  The reproduction, distribution and utilization of this file as
#  well as the communication of its contents to others without express
#  authorization is prohibited. Offenders will be held liable for the
#  payment of damages. All rights reserved in the event of the grant
#  of a patent, utility model or design.
# ==============================================================================
import os
import sys
sys.path.insert(0, os.path.abspath('../../deployment_artifacts/xaimodules/'))

project = 'docs4xai'  # pylint: disable=C0103
copyright = '2024, OnePMT'  # pylint: disable=C0103, W0622
author = 'OnePMT'  # pylint: disable=C0103
release = '1.0.0'  # pylint: disable=C0103

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'm2r2',
    'sphinxcontrib.drawio',
]

templates_path = ['_templates']
# exclude_patterns = []
sphinx_rtd_size_width = '90%'  # pylint: disable=C0103

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # pylint: disable=C0103
html_static_path = ['_static']
source_suffix = ['.rst', '.md']
pygments_style = 'sphinx'  # pylint: disable=C0103
