# -*- coding: utf-8 -*-

import sys
import os

file = sys.argv[-1]

sh = f"""#! /bin/bash
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -o %J.out
#SBATCH -G 1

python -u ./{file}
"""

with open("_run.sh", "wb") as f:
    f.write(sh.encode("utf-8"))

os.system("sbatch _run.sh")