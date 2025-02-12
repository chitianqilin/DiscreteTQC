
envs = """Ant-v5 5
HalfCheetah-v5 5
Hopper-v5 5
Humanoid-v5 5
Walker2d-v5 5
Swimmer-v5 0.3
"""

# envs = """
# HalfCheetahmujocoEnv-v0
# AntmujocoEnv-v0
# HoppermujocoEnv-v0
# Walker2DmujocoEnv-v0
# HumanoidmujocoEnv-v0
# """

import os
import shutil

if os.path.exists("./ours"):
    shutil.rmtree("./ours")

os.mkdir("./ours")

run = """#! /bin/bash
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -o %J.out
#SBATCH -G 1



"""

i = 0

def time():
    global i
    i += 1
    return str(i).zfill(2)

for line in sorted(envs.splitlines()):
    env, rn = line.split()
    exp = f"{time()}-{env}"
    os.mkdir(f"./ours/{exp}")
    with open(f"./ours/{exp}/{exp}-run.sh", "wb") as f:
        code = run + f"python -u ./ours-mujoco.py {env} {rn}"
        f.write(code.encode("utf-8"))
    with open(f"./ours-mujoco.py", "rb") as f:
        code = f.read().decode("utf-8")
    with open(f"./ours/{exp}/ours-mujoco.py", "wb") as f:
        f.write(code.encode("utf-8"))

"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/home/zhangjundong/.mujoco/mujoco210/bin"