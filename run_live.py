import subprocess

addr = open('config/address.txt').readline().strip()
subprocess.check_call(['python', 'motiondetector.py', addr])