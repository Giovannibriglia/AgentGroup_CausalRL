import os
import sys

# Get the absolute path of the root directory of your repository
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Append the root directory to the Python module search path
sys.path.append(repo_root)

# Now you can import global_variables.py
import global_variables

# Access global variables defined in global_variables.py
print(global_variables.GLOBAL_PATH_REPO)

