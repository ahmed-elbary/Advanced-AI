import subprocess
import sys

# List of libraries to install
libraries = [
    "pandas",
    "pgmpy",
    "sklearn",
    "numpy",
    "time",
    'matplotlib',
    'networkx'
]

# Install each library
for library in libraries:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library])
