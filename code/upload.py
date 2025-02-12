
import pexpect
import os
import sys

# 上传脚本，在linux python3环境下运行，python upload.py xxxx 可以把xxxx上传到超算

from info import *

if len(sys.argv) < 2:
    print("error")
    exit()

# 获取你要上传的文件的路径
file = sys.argv[-1]
if file.startswith("./"):
    file = file[2:]
if file.endswith("/"):
    file = file[:-1]
# 可以是目录也可以是文件
isdir = os.path.isdir(sys.argv[-1])

bash = pexpect.spawn(f"zsh", timeout=60*60, encoding="utf-8")
bash.logfile = sys.stdout

# 登录超算并删除有冲突的文件
bash.sendline(f"ssh -J {user}@172.16.108.134 {user}@192.168.10.15")
for j in range(2):
    i = bash.expect(["yes", "password"])
    if i == 0:
        bash.sendline("yes")
        bash.expect("password")
    bash.sendline(password)
bash.expect(wait_remote)

bash.sendline(f"mkdir ./{project}")
bash.expect(wait_remote)

# 删除有冲突的文件
flag = "-rf" if isdir else ""
bash.sendline(f"rm {flag} ./{project}{file}")
bash.expect(wait_remote)
bash.sendline("logout")
bash.expect(wait_local)

# 上传文件
flag = "-r" if isdir else ""
bash.sendline(f"scp -o 'ProxyJump {user}@172.16.108.134' {flag} ./{file} {user}@192.168.10.15:~/{project}{file}")
for j in range(2):
    i = bash.expect(["yes", "password"])
    if i == 0:
        bash.sendline("yes")
        bash.expect("password")
    bash.sendline(password)
    
bash.expect(wait_local)
bash.logfile = None
