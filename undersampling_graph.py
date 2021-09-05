tm_overall_mape = [9.106358522, 10.27132887, 26.42329126, 145.9317466, 158.9987585]
tm_t1_mape = [14.4531295, 16.20711099, 30.36851269, 71.93456924, 83.71295191]
tm_t2_mape = [3.759587546, 4.335546751, 22.47806983, 219.928924, 234.284565]

blip_overall_mape = [9.06120182, 10.29918799, 11.51969989, 14.05853326, 18.88368581]
blip_t1_mape = [14.35537621, 16.37565666, 18.32056357, 22.3388962, 29.82925264]
blip_t2_mape = [3.767027434, 4.22271932, 4.718836212, 5.778170319, 7.938118991]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
fig.subplots_adjust(wspace=0.3)
ax.plot(tm_t1_mape)
ax.plot(tm_t2_mape)
ax.plot(blip_t1_mape)
ax.plot(blip_t2_mape)
ax.set_ylabel("MAPE (%)", family='Arial', fontsize=15)
ax.set_xlabel("Cartesian Undersampling Ratio", family='Arial', fontsize=15)
ax.set_xticklabels(["1/1", "1/2", "1/4", "1/8", "1/16"])


plt.show()
# cmap = None
# im = ax[0][0].matshow(actual_t1_map, vmin=0, vmax=3000, cmap=cmap)
# ax[0][0].title.set_text("True T1")
# ax[0][0].set_xticks([]), ax[0][0].set_yticks([])
# # https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
# divider = make_axes_locatable(ax[0][0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im, cax=cax, shrink=0.8)