import sys
import os

# assuming all main files and imports are one level above "tests" subfolder
main_repo_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# if not present, add it
if main_repo_root not in sys.path:
    sys.path.insert(0, main_repo_root)