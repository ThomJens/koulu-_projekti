import sys
import subprocess
import pkg_resources
import warnings
warnings.filterwarnings('ignore')

def installation() -> None:
    """"Install missing libraries."""
    libraries = {'pandas', 'seaborn', 'matplotlib', 'numpy', 'psycopg2-binary', 'sqlalchemy', 'scikit-learn', 'scipy'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = libraries - installed

    # False if list is empty. Installs the missing python libraries.
    if missing:
        python = sys.executable
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing], stdout = subprocess.DEVNULL)
        

