import pexpect
import sys

# 登录脚本，在linux python3环境下运行，python login.py 可以登录超算

from info import *

# 登录跳板机和超算
bash = pexpect.spawn(f"ssh -J {user}@172.16.108.134 {user}@192.168.10.15", 
    timeout=60*60,
    encoding="utf-8")
# 将信息输出到屏幕
bash.logfile = sys.stdout
# 输入两次密码
for j in range(2):
    i = bash.expect(["yes", "password"])
    if i == 0:
        bash.sendline("yes")
        bash.expect("password")
    bash.sendline(password)
bash.expect(wait_remote)

# 设置超算代理节点，确保可以使用外网
bash.sendline("export http_proxy=http://192.168.10.22:3128;export https_proxy=http://192.168.10.22:3128")
bash.expect(wait_remote)

# 配置module路径，确保可以使用安装好的模块
bash.sendline("export MODULEPATH=/share/app_share/modulefiles")
bash.expect(wait_remote)

bash.sendline("module avail")
bash.expect(wait_remote)
bash.sendline("module load Anaconda/mini3-23.1.0")
bash.expect(wait_remote)
bash.sendline("module load cuda/12.1.0")
bash.expect(wait_remote)
bash.sendline("conda activate mujoco_env")
bash.expect(wait_remote)

bash.sendline(f"cd ./{project}")
bash.expect(wait_remote)


# 转为交互模式
bash.logfile = None
bash.interact()

