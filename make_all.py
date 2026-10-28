"""Run every part in order, or only the scripts named: the datasets go to data/, figures and numbers to out/."""
import os
import subprocess
import sys

SCRIPTS = ['simulate.py', 'perceptron.py', 'kernels.py', 'augment.py', 'shuffle.py', 'dimensions.py']
for script in sys.argv[1:] or SCRIPTS:
    print('running', script, flush=True)
    subprocess.run([sys.executable, script], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
