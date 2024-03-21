import os, sys
sys.path.append(os.path.abspath("error_concealment_interface"))
from error_concealment_interface import ec_driver

ec_driver.run_expr("INDEX.txt", "results/error_concealment")
