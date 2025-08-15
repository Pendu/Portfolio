# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Personal Website'
copyright = '2025, Abhijeet Pendyala'
author = 'Abhijeet Pendyala'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "use_edit_page_button": False,
    "logo": {
        "text": "Home",
    },
    "header_links_before_dropdown": 6,
}

html_static_path = ['_static']

# Set HTML titles to override default "documentation" references
html_title = 'Personal Website'
html_short_title = 'Abhijeet Pendyala'

# Add logo for home button - using a professional home icon
html_logo = '_static/home_icon.svg'

# Add custom CSS file - path relative to html_static_path directory
html_css_files = [
    "_static/custom.css",
]

# Furo theme options - simplified and well-supported
# html_theme_options = {
#     "light_css_variables": {
#         "color-brand-primary": "#007acc",
#         "color-brand-content": "#007acc",
#     },
#     "dark_css_variables": {
#         "color-brand-primary": "#007acc",
#         "color-brand-content": "#007acc",
#     },
#     "navigation_with_keys": True,
#     "source_repository": "https://github.com/yourusername/portfolio_2025",
#     "source_branch": "main",
#     "source_directory": "source/",
# }

# Remove sidebars for a cleaner look and better image floating
html_sidebars = {"**": []}

# Additional theme options for better layout
html_theme_options = {
    "use_edit_page_button": False,
    "logo": {
        "text": "Home",
    },
    "header_links_before_dropdown": 6,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
}

# -- Options for bibtex ----------------------------------------
bibtex_bibfiles = ["refs.bib"]