
# 你在超算上的账号密码
user = "zhangjundong"
password = "5Sa!cu!Vg9q4"
# 你在超算上的称呼
wait_remote = r"zhangjundong@manage"
# 你在本地电脑的称呼
wait_local = r"zhangjundong@zhangjundongdeMacBook-Air"

"""
ssh -J zhangjundong@172.16.108.134 zhangjundong@192.168.10.15 -L 8888:localhost:8888 -L 6007:localhost:6006
"""

import os

project = os.path.split(os.path.dirname(__file__))[-1] + "/"
