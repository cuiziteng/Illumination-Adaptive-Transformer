_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_wtih_IAT_MIT5K.py',
    '../_base_/datasets/ACDC_night.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]