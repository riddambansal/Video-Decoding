import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Replace 'your_file.csv' with the actual file path
file_path = 'ss3_ALL.csv'
# Load the CSV file with 23 header lines
data_1 = pd.read_csv(file_path, header=10)
data = data_1.iloc[:-3]
data.drop(data.index[1])
#%% get Data info
sampling_interval = data["TIME"][1]-data["TIME"][0]
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
#%% extract pixel data from [ITU-R BT.656 Tx CONFIGURATION] 8 bit data
start_y_pos = 1
y = CH8_dec[start_y_pos::2] #as 4:2:2 is rcvd
start_cb_pos = 0
cb = CH8_dec[start_cb_pos::4] #as 4:2:2 is rcvd
start_cr_pos = 2
cr = CH8_dec[start_cr_pos::4] #as 4:2:2 is rcvd
#%%
YCbCr = []
RGB = []

for i in np.arange(len(y)-1):
    # print(i,i//2,i//2)
    # https://en.wikipedia.org/wiki/YCbCr
    Y = y[i]
    Cb = cb[i//2]
    Cr = cr[i//2]
    YCbCr.append([Y, Cb, Cr])
    #ITU-R BT.656 conversion factors https://techdocs.altium.com/display/FPGA/BT656+-+Color+Conversion
    # R = Y+Cb
    # G = Y
    # B = Y+Cr
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    RGB.append([R,G,B])
    i += 1
YCbCr = np.array(YCbCr).reshape(y.shape[0]-1,3)
RGB = np.array(RGB).reshape(y.shape[0]-1,3)
fig, ax = plt.subplots(1, 1)
ax.plot(y[0::2], 'k')
ax.plot(cb, 'b')
ax.plot(cr, 'r')
plt.show()

#%%
# def ycbcr_to_rgb(y, cb, cr):
#     # Convert YCbCr to RGB
#     # R = y + 1.13983 * (cr - 128)
#     # G = y - 0.39465 * (cb - 128) - 0.58060 * (cr - 128)
#     # B = y + 2.03211 * (cb - 128)
#     R = 128
#     G = 128
#     B = 128

#     # Ensure RGB values are within the valid range [0, 255]
#     R = min(max(R, 0), 255)
#     G = min(max(G, 0), 255)
#     B = min(max(B, 0), 255)

#     return int(R), int(G), int(B)

# # Example usage:
# y_value = 128  # Replace with your Y value (0-255)
# cb_value = 100  # Replace with your Cb value (0-255)
# cr_value = 150  # Replace with your Cr value (0-255)

# rgb = ycbcr_to_rgb(y_value, cb_value, cr_value)
# print("RGB:", rgb)

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

