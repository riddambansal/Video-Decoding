import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Replace 'your_file.csv' with the actual file path
file_path = 'ss2_ALL.csv'
# Load the CSV file with 23 header lines
data = pd.read_csv(file_path, header=10)
# data = data_1.iloc[:-3]
# data = data.drop(data.index[1])
#%% get Data info
sampling_interval = data["TIME"][10]-data["TIME"][9]
MSPS = np.ceil((1/sampling_interval)/1e6)
d = data.iloc[:, -8:].values# [:, ::-1] MSB Corrected not required
num_samples = d.shape[0]
duration_data_us = num_samples*sampling_interval*1e6
#%% prepare clock 
# u  = yuv[0];
# y1 = yuv[1];
# v  = yuv[2];
# y2 = yuv[3];

llc = data["CH2"].values
th = 1.5  # Threshold value
# Apply the threshold
clk = (llc > th).astype(int)

# Find the rising edges in the thresholded_array
rising_edges_index = np.where(np.diff(clk) > 0)[0]

# Extract values from another_array at the rising edge indexes
resampled_CH8 = d[rising_edges_index,:]
t =  data["TIME"][rising_edges_index]
CH8_dec = np.packbits(resampled_CH8, axis=1)
CH8_dec = CH8_dec[3:]
#%% extract pixel data from [ITU-R BT.656 Tx CONFIGURATION] 8 bit data
start_pos = 0
start_y_pos = start_pos + 1
y = CH8_dec[start_y_pos::2] #as 4:2:2 is rcvd
start_cb_pos = start_pos + 0
cb = CH8_dec[start_cb_pos::4] #as 4:2:2 is rcvd
start_cr_pos = start_pos + 2
cr = CH8_dec[start_cr_pos::4] #as 4:2:2 is rcvd
#%%
YCbCr = []
RGB = []
nof_pix = len(y)-2
for i in np.arange(nof_pix):
    # print(i,i//2,i//2)
    # https://en.wikipedia.org/wiki/YCbCr
    Y = y[i]
    Cb = cb[i//2]
    Cr = cr[i//2]
    YCbCr.append([Y, Cb, Cr])
    #ITU-R BT.656 conversion factors https://techdocs.altium.com/display/FPGA/BT656+-+Color+Conversion
    R = int(np.floor(Y + 1.402*Cr))
    G = int(np.floor(Y - 0.344*Cb - 0.714*Cr))
    B = int(np.floor(Y + 1.772*Cb))
    RGB.append([R,G,B])
    i += 1
YCbCr = np.array(YCbCr).reshape(nof_pix,3)
RGB = np.array(RGB).reshape(nof_pix,3)
fig, ax = plt.subplots(1, 1)
# ax.plot(y[0::2], 'k')
# ax.plot(cb, 'b')
# ax.plot(cr, 'r')
ax.plot(y, 'k')
ax.plot(cb.repeat(2), 'b')
ax.plot(cr.repeat(2), 'r')
plt.show()
#%%
YCbCr = []
RGB = []
nof_pix = len(y)-2
for i in np.arange(nof_pix):
    # print(i,i//2,i//2)
    # https://en.wikipedia.org/wiki/YCbCr
    Y = y[i]
    Cb = cb[i//2]
    Cr = cr[i//2]
    YCbCr.append([Y, Cb, Cr])
    #ITU-R BT.656 conversion factors https://techdocs.altium.com/display/FPGA/BT656+-+Color+Conversion
    R = Cr+Y
    G = Y
    B = Cb+Y
    RGB.append([R,G,B])
    i += 1
YCbCr = np.array(YCbCr).reshape(nof_pix,3)
RGB = np.array(RGB).reshape(nof_pix,3)
fig, ax = plt.subplots(1, 1)
# ax.plot(y[0::2], 'k')
# ax.plot(cb, 'b')
# ax.plot(cr, 'r')
ax.plot(y, 'k')
ax.plot(cb.repeat(2), 'b')
ax.plot(cr.repeat(2), 'r')
plt.show()


#%%
image_height = 1
image_width = RGB.shape[0]
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
pixel_line = RGB
# Create a figure and axis for the plot
fig, ax = plt.subplots(1, 1, figsize=(len(pixel_line), 1))

# Display the line of pixels as an image
ax.imshow([pixel_line], aspect='auto')

# Remove axis labels and ticks
ax.axis('off')

# Show the plot
plt.show()

# #%%
# # Constants
# width = 720  # Width of the frame (pixels)
# height = 576  # Height of the frame (pixels)
# num_bars = 8  # Number of grayscale bars
# bar_width = width // num_bars  # Width of each bar

# # Create an empty frame (all black)
# frame = np.zeros((height, width), dtype=np.uint8)

# # Generate grayscale bars
# for i in range(num_bars):
#     # Calculate the brightness value for the bar (gradually increasing)
#     brightness = int((i / num_bars) * 255)
    
#     # Set the pixels in the current bar to the calculated brightness
#     frame[:, i * bar_width : (i + 1) * bar_width] = brightness

# # Display the frame using Matplotlib
# plt.figure()
# plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.show()
# # Create R, G, and B matrices by replicating the grayscale frame
# R_matrix = frame.copy()
# G_matrix = frame.copy()
# B_matrix = frame.copy()

# # Display the grayscale R, G, and B matrices
# print("R Matrix:")
# print(R_matrix)

# print("\nG Matrix:")
# print(G_matrix)

# print("\nB Matrix:")
# print(B_matrix)
