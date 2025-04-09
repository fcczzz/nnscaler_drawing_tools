import scienceplots
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import numpy as np

plt.style.use(['bright'])

# plt.style.use(['science', 'ieee', 'std-colors'])
gpu_nums = [4, 8, 16, 32]
megatron_lm = [20, 60, 0, 180]

alpa = [25, 55, 80, 175]
deepspeed = [15, 50, 78, 250]
nnscaler = [35, 70, 120, 380]
nnscaler_new = []

# 设置柱宽和位置
bar_width = 0.2
x = np.arange(len(gpu_nums))

# 绘图
fig, ax = plt.subplots(figsize=(8,4))

ax.bar(x - 1.5 * bar_width,
       megatron_lm,
       bar_width,
       label='Megatron-LM',
       color='#fe7f10',
       hatch='+',
       edgecolor='white')
ax.bar(x - 0.5 * bar_width,
       alpa,
       bar_width,
       label='Alpa',
       color='#d52628',
       hatch='x',
       edgecolor='white')
ax.bar(x + 0.5 * bar_width,
       deepspeed,
       bar_width,
       label='DeepSpeed',
       color='#2ba02d',
       hatch='o',
       edgecolor='white')
ax.bar(x + 1.5 * bar_width,
       nnscaler,
       bar_width,
       label='nnScaler',
       color='#1f77b4',
       edgecolor='white')

# 标签和图例
ax.set_xlabel('GPU Number', fontsize=18)
ax.set_ylabel('Throughput (TFLOPS)', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(gpu_nums, fontsize=18)

ax.yaxis.set_major_locator(MultipleLocator(100))
ax.grid(axis='y', linewidth=0.7, color='gray', alpha=0.4)

plt.rcParams.update({'legend.fontsize': 18})
ax.set_ylim(0, 400)
for label in ax.get_yticklabels():
    label.set_fontsize(18)

ax.legend(framealpha=0) 

plt.tight_layout()
plt.show()
