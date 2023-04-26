#! /usr/bin/env python2

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


mental_demand1 = np.array([3.4, 2.6, 4.1])
mental_demand2 = np.array([2.1, 3.1, 3.1])
phys_demand1 = np.array([1.9, 4.6, 4.1])
phys_demand2 = np.array([2.3, 5, 2.9])
temp_demand1 = np.array([1.7, 2.8, 4])
temp_demand2 = np.array([2.6, 3.6, 3])

fig1, ax1 = plt.subplots(1, 3, figsize=(6, 3))
fig1.tight_layout()

ax1[0].boxplot([mental_demand1, mental_demand2], patch_artist=False,
                positions=[1,1.5],
                widths=[0.5, 0.5],
                labels=['E1', 'E2'])
ax1[0].set_ylabel('mental demand')
ax1[0].set_ylim(0, 8)

ax1[1].boxplot([phys_demand1, phys_demand2], patch_artist=False,
                positions=[1,1.5],
                widths=[0.5, 0.5],
                labels=['E1', 'E2'])
ax1[1].set_ylabel('physical demand')
ax1[1].set_ylim(0, 8)

ax1[2].boxplot([temp_demand1, temp_demand2], patch_artist=False,
                positions=[1,1.5],
                widths=[0.5, 0.5],
                labels=['E1', 'E2'])
ax1[2].set_ylabel('temp_demand')
ax1[2].set_ylim(0, 8)

fig1.savefig('/home/i2r2020/Documents/huanfei/bo_data/workload1.tif', dpi=300, bbox_inches="tight")


fig2, ax2 = plt.subplots(1, 3, figsize=(6, 3))
fig2.tight_layout()

performance1 = np.array([3.9, 5.8, 5.6])
performance2 = np.array([3, 5.4, 6.5])
effort1 = np.array([2.6, 5.2, 4])
effort2 = np.array([2.5, 3.6, 3.3])
frustration1 = np.array([3.8, 2.9, 3.6])
frustration2 = np.array([2.3, 2.1, 1.8])

ax2[0].boxplot([performance1, performance2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['E1', 'E2'])
ax2[0].set_ylabel('performance')
ax2[0].set_ylim(0, 8)
ax2[1].boxplot([effort1, effort2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['E1', 'E2'])
ax2[1].set_ylabel('effort')
ax2[1].set_ylim(0, 8)
ax2[2].boxplot([frustration1, frustration2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['E1', 'E2'])
ax2[2].set_ylabel('frustration')
ax2[2].set_ylim(0, 8)
fig2.savefig('/home/i2r2020/Documents/huanfei/bo_data/workload2.tif', dpi=300, bbox_inches="tight")

fig3, ax3 = plt.subplots(1, 3, figsize=(6, 3))
fig3.tight_layout()

failure1 = np.array([5, 4, 6])
failure2 = np.array([3, 2, 3])
contact_lost1 = np.array([8, 6, 6])
contact_lost2 = np.array([3, 4, 4])

ax3[0].boxplot([failure1, failure2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['E1', 'E2'])
ax3[0].set_ylabel('Failure')
ax3[0].set_ylim(0, 10)
ax3[1].boxplot([contact_lost1, contact_lost2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['E1', 'E2'])
ax3[1].set_ylabel('Contact loss')
ax3[1].set_ylim(0, 10)
fig3.savefig('/home/i2r2020/Documents/huanfei/bo_data/workload3.tif', dpi=300, bbox_inches="tight")

plt.show()