
'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.
The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
'''

import tensorflow as tf
import glob
import os
import pandas as pd
from scipy import interpolate
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os

from tensorboard.backend.event_processing import event_accumulator


def convert_to_csv():
    # Get all event* runs from logging_dir subdirectories
    logging_dir = './logs'
    event_paths = glob.glob(os.path.join(logging_dir, "*", "event*"))

    # Extraction function
    def sum_log(path):
        runlog = pd.DataFrame(columns=['metric', 'value'])
        try:
            for e in tf.train.summary_iterator(path):
                for v in e.summary.value:
                    r = {'metric': v.tag, 'value': v.simple_value}
                    runlog = runlog.append(r, ignore_index=True)

        # Dirty catch of DataLossError
        except:
            print('Event file possibly corrupt: {}'.format(path))
            return None

        runlog['epoch'] = [item for sublist in [[i] * 5 for i in range(0, len(runlog) // 5)] for item in sublist]

        return runlog

    # Call & append
    all_log = pd.DataFrame()
    for path in event_paths:
        log = sum_log(path)
        if log is not None:
            if all_log.shape[0] == 0:
                all_log = log
            else:
                all_log = all_log.append(log)

    # Inspect
    print(all_log.shape)
    all_log.head()

    # Store
    all_log.to_csv('all_training_logs_in_one_file.csv', index=None)


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)#, arr.max(axis=-1), arr.min(axis=-1)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_processed_records(f='file', window=int(1e3)):
    # load log data
    ea = event_accumulator.EventAccumulator(f)
    ea.Reload()
    print(ea.scalars.Keys())

    val_psnr = ea.scalars.Items('rollout/ep_imitation_mean')

    # print(val_psnr.step)

    steps = []
    rewards = []
    for v in val_psnr:
        steps.append(v.step)
        rewards.append(v.value)

    steps = np.array(steps)
    rewards = np.array(rewards)

    new_steps = np.linspace(0, steps[-1], steps[-1])

    #func = interpolate.interp1d(steps, rewards, kind='cubic')
    func = interpolate.interp1d(steps, rewards, axis=0, fill_value="extrapolate")

    new_rewards = func(new_steps)
    new_rewards = smooth(new_rewards, window)

    return (new_steps, new_rewards)

train_records = []
for root, dirnames, filenames in os.walk('tmp/pwil2'):
    for filename in fnmatch.filter(filenames, '*'):
        log_file_name = os.path.join(root, filename)
        if "by_none" in log_file_name and "BipedalWalker-v3" in log_file_name:
            try:
                print(log_file_name)
                new_steps, new_rewards = get_processed_records(log_file_name)
                train_records.append(new_rewards)
            except:
                True

fig, ax = plt.subplots()

mean_rewards, error= tolerant_mean(train_records)
lower_bound =  mean_rewards - 100
upper_bound =  mean_rewards + error
print(error)
plt.plot(np.arange(len(mean_rewards)) + 1, mean_rewards, color='green')
# for t in train_records:
#    plt.plot(np.arange(len(t)) + 1, t, color='blue')

ax.fill_between(len(mean_rewards), lower_bound, upper_bound, facecolor='green', alpha=1, label='1 sigma range')

#plt.plot(new_steps, smooth(new_rewards,10000))
plt.show()
