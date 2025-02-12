
import pexpect
import os
import sys

# 下载脚本，在linux python3环境下运行，python upload.py xxxx 可以把xxxx下载到本地

from info import *

if len(sys.argv) < 2:
    print("error")
    exit()

file = sys.argv[-1]
if file.startswith("./"):
    file = file[2:]
if file.endswith("/"):
    file = file[:-1]
isdir = os.path.isdir(sys.argv[-1])

bash = pexpect.spawn(f"zsh", encoding="utf-8", timeout=60*60*24)
bash.logfile = sys.stdout

# 删除本地有冲突的文件
bash.expect(wait_local)
flag = "-rf" if isdir else ""
bash.sendline(f"rm {flag} ./{file}")
bash.expect(wait_local)

# 从超算下载目标文件下来
flag = "-r" if isdir else ""
bash.sendline(f"scp -o 'ProxyJump {user}@172.16.108.134' {flag} {user}@192.168.10.15:~/{project}{file} ./{file}")
for j in range(2):
    i = bash.expect(["yes", "password"])
    if i == 0:
        bash.sendline("yes")
        bash.expect("password")
    bash.sendline(password)
    
bash.expect(wait_local)
bash.logfile = None
