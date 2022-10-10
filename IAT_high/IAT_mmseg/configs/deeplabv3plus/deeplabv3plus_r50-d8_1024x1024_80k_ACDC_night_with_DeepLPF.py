_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/ACDC_night_with_DeepLPF.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
