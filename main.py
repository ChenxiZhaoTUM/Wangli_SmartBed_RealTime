import re
import torch
import numpy as np


def process_pressure_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AA23") and values.endswith("55"):
            pres_hex_values = values[4:-2]

            if len(pres_hex_values) == 64:
                pres_decimal_arr = [4095 - int(pres_hex_values[i + 2:i + 4] + pres_hex_values[i:i + 2], 16) for i in
                                    range(0, len(pres_hex_values), 4)]
                return time, pres_decimal_arr

    return time, []


def process_sleep_values(line):
    time_match = re.search(r'\[(.*?)\]', line)
    if time_match:
        time = time_match.group(1)
        values = line.split(']')[-1]

        if values.startswith("AB11") and values.endswith("55"):
            pres_hex_values = values[4:-2]

            if len(pres_hex_values) == 28:
                processed_hex_values = pres_hex_values[:12] + ''.join(
                    [pres_hex_values[i + 2:i + 4] + pres_hex_values[i:i + 2] for i in range(12, 20, 4)])
                processed_hex_values += pres_hex_values[20:22] + pres_hex_values[24:26] + pres_hex_values[
                                                                                          22:24] + pres_hex_values[
                                                                                                   26:28]

                pres_decimal_arr = [int(processed_hex_values[i:i + 2], 16) for i in range(0, 12, 2)] + [
                    int(processed_hex_values[i:i + 4], 16) for i in range(12, 24, 4)] + [
                                       int(processed_hex_values[24:26], 16), int(processed_hex_values[26:28], 16)]
                return time, pres_decimal_arr

    return time, []


# def change_dimension(input_data, input_sleep_data):
#     new_input_data = [[[0 for _ in range(64)] for _ in range(32)] for _ in range(12)]
#
#     for ch in range(11):
#         for i in range(32):
#             for j in range(64):
#                 new_input_data[ch][i][j] = input_sleep_data[ch]
#
#     for j in range(16):
#         for i in range(32):
#             for k in range(4):
#                 new_input_data[11][i][j * 4 + k] = input_data[j]
#
#     return new_input_data

def change_dimension(input_data, input_sleep_data):
    new_input_data = torch.zeros(12, 32, 64)

    for ch in range(11):
        new_input_data[ch, :, :] = torch.tensor(input_sleep_data[ch])  # input_sleep_data to 11 channels

    for j in range(16):
        new_input_data[11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data[j])

    return new_input_data


def print_3d_vector(vec):
    for i, channel in enumerate(vec):
        print(f"Channel {i}:")
        for row in channel:
            print(" ".join(map(str, row)))
        print()


# Example usage
line0 = "[16:07:59.569]AA23F40FFF0FFF0F3B0FFF0FFF0FFF0F8E0F280D64086B0CBC05140B7F0C4F0AFF0F55"
time, values = process_pressure_values(line0)
# print(values)
pressure_map = {time: values}

line0_sleep = "[16:07:59.789]AB11450C00000001861B060000710A1955"
time_sleep, values_sleep = process_sleep_values(line0_sleep)
sleep_map = {time_sleep: values_sleep}

new_input_data_tensor = change_dimension(values, values_sleep)
print(new_input_data_tensor)
# print_3d_vector(new_input_data)
