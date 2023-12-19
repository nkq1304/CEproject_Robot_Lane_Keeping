import numpy as np
import matplotlib.pyplot as plt

backend_node_times = np.load('backend_node_times.npy')
control_lane_times = np.load('control_lane_times.npy')
process_image_times = np.load('process_image_times.npy')

backend_node_times = backend_node_times[5:]
control_lane_times = control_lane_times[5:]
process_image_times = process_image_times[5:]

# multiply by 1000 to get ms
backend_node_times *= 1000
control_lane_times *= 1000
process_image_times *= 1000
total_times = backend_node_times + process_image_times

errors = np.load('errors.npy')

plt.plot(np.convolve(errors, np.ones(10)/10, mode='valid'))
plt.xlabel('frame')
plt.ylabel('error')
plt.axhline(y=0, color='r', linestyle='-')

plt.show()

# plt.plot(np.convolve(backend_node_times, np.ones(50)/50, mode='valid'), label='Backend')
# plt.plot(np.convolve(process_image_times, np.ones(50)/50, mode='valid'), label='TwinLiteNet')
plt.plot(np.convolve(total_times, np.ones(50)/50, mode='valid'))

plt.legend()
plt.xlabel('frame')
plt.ylabel('ms')

plt.show()