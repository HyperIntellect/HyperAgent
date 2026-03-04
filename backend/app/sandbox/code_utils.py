"""Shared code analysis utilities for sandbox executors.

Provides package detection and import injection logic that is shared across
all sandbox providers (E2B, BoxLite, etc.). These functions have no
provider-specific dependencies.
"""

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# Map import names to pip package names
PACKAGE_MAPPING = {
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "numpy": "numpy",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "plotly": "plotly",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "flask": "flask",
    "fastapi": "fastapi",
    "sqlalchemy": "sqlalchemy",
    "psycopg2": "psycopg2-binary",
    "pymongo": "pymongo",
    "redis": "redis",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "boto3": "boto3",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "networkx": "networkx",
    "sympy": "sympy",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

# Usage patterns that indicate a package is needed (pattern -> pip package name)
USAGE_PATTERNS = {
    r"\bpd\.": "pandas",
    r"\bnp\.": "numpy",
    r"\bplt\.": "matplotlib",
    r"\bsns\.": "seaborn",
    # pandas Excel operations implicitly require openpyxl
    r"read_excel\s*\(": "openpyxl",
    r"\.to_excel\s*\(": "openpyxl",
    r"\bExcelFile\s*\(": "openpyxl",
    r"\bExcelWriter\s*\(": "openpyxl",
}


def detect_required_packages(code: str) -> list[str]:
    """Detect Python packages required by the code.

    Analyzes import statements and usage patterns to determine which
    packages need to be installed.

    Args:
        code: Python code to analyze

    Returns:
        List of package names to install via pip
    """
    required_packages: set[str] = set()

    # Check for explicit imports
    # Match: import foo, from foo import bar, import foo.bar
    import_pattern = r"(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    for match in re.finditer(import_pattern, code):
        module_name = match.group(1)
        if module_name in PACKAGE_MAPPING:
            required_packages.add(PACKAGE_MAPPING[module_name])

    # Check for usage patterns (handles cases where import is missing)
    for pattern, package in USAGE_PATTERNS.items():
        if re.search(pattern, code):
            required_packages.add(package)

    return list(required_packages)


def inject_python_imports(code: str) -> str:
    """Inject common Python imports if not present in the code.

    This helps prevent NameError when the LLM generates code that uses
    common libraries (plt, pd, np, sns) without importing them.

    Also ensures matplotlib uses non-interactive backend for headless environments.

    Args:
        code: Original Python code

    Returns:
        Code with necessary imports prepended
    """
    # Common imports for data analysis
    import_mappings = {
        "pd.": "import pandas as pd",
        "pd,": "import pandas as pd",
        "pandas.": "import pandas as pd",
        "np.": "import numpy as np",
        "np,": "import numpy as np",
        "numpy.": "import numpy as np",
        "plt.": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
        "matplotlib.": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
        "sns.": "import seaborn as sns",
        "seaborn.": "import seaborn as sns",
    }

    missing_imports: list[str] = []

    for usage, import_statement in import_mappings.items():
        # Check if the usage exists in code but import is missing
        if usage in code and import_statement not in code:
            # Check if this is matplotlib-related
            if "matplotlib" in import_statement:
                # Check if matplotlib backend is already configured
                if "matplotlib.use" not in code and "matplotlib\nmatplotlib.use" not in "\n".join(
                    missing_imports
                ):
                    if import_statement not in missing_imports:
                        missing_imports.append(import_statement)
            else:
                # For non-matplotlib imports, check variations
                base_module = import_statement.split()[1]
                if f"import {base_module}" not in code and f"from {base_module}" not in code:
                    if import_statement not in missing_imports:
                        missing_imports.append(import_statement)

    # If code already has matplotlib import but not backend, inject backend before it
    if ("import matplotlib" in code or "from matplotlib" in code) and "matplotlib.use" not in code:
        # Prepend backend setting
        backend_setup = "import matplotlib\nmatplotlib.use('Agg')\n\n"
        logger.info("injected_matplotlib_backend")
        code = backend_setup + code

    if missing_imports:
        imports_block = "\n".join(missing_imports) + "\n\n"
        logger.info("injected_python_imports", imports=missing_imports)
        return imports_block + code

    return code
