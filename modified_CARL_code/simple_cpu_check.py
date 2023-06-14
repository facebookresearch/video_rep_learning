import psutil
import torch
import os
print('psutil CPU: ' + str(psutil.Process().cpu_num()))
print('psutil CPU Count: ' + str(psutil.cpu_count()))
print('psutil CPU utilization:')
print(psutil.cpu_percent(percpu=True))
print('---')
print('os.cpu_count()')
print(os.cpu_count())
print('---')
print('torch.get_num_threads()')
print(torch.get_num_threads())