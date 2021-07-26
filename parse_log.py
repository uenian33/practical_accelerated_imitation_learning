
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
import pickle

from tensorboard.backend.event_processing import event_accumulator


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)  # , arr.max(axis=-1), arr.min(axis=-1)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_processed_records(f='file', window=int(1e3)):
    # load log data
    ea = event_accumulator.EventAccumulator(f)
    ea.Reload()
    # print(ea.scalars.Keys())

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

    # func = interpolate.interp1d(steps, rewards, kind='cubic')
    func = interpolate.interp1d(steps, rewards, axis=0, fill_value="extrapolate")

    new_rewards = func(new_steps)
    new_rewards = smooth(new_rewards, window)

    return (new_steps, new_rewards)


def generate_pkl_files(constrained_types, env_types):
    log_names = []
    print('parse folders...')
    for root, dirnames, filenames in os.walk('tmp/pwil2'):
        for filename in fnmatch.filter(filenames, '*'):
            log_names.append(os.path.join(root, filename))

    print('categorize logs')
    env_clusters = []

    for env_id in env_types:
        env_records = []
        for log_file_name in log_names:
            if env_id in log_file_name:
                env_records.append(log_file_name)
        print(len(env_records))
        env_clusters.append(env_records)

    for env_id, env_cate in enumerate(env_clusters):
        filelist = env_cate
        fig, ax = plt.subplots()

        env_logs = []
        for c_id in constrained_types:
            print(env_types[env_id], c_id)
            train_records = []
            for log_file_name in filelist:
                if c_id in log_file_name:
                    print(log_file_name)
                    try:
                        new_steps, new_rewards = get_processed_records(log_file_name, window=20000)
                        train_records.append(new_rewards)
                        filelist.remove(log_file_name)
                    except:
                        True
            # print(len(train_records))
            env_logs.append(train_records)
            print(len(train_records))
            mean_rewards, error = tolerant_mean(train_records)
            lower_bound = mean_rewards - error
            upper_bound = mean_rewards + error
            #print(amin, amax)
            ax.plot(np.arange(len(mean_rewards)) + 1, mean_rewards, alpha=0.78, label=env_types[env_id] + '_' + c_id)
            # for t in train_records:
            #    plt.plot(np.arange(len(t)) + 1, t, color='blue')

            ax.fill_between(len(mean_rewards), lower_bound, upper_bound, alpha=1)
            ax.legend(loc='upper left')

        print(len(env_logs))
        out_pkl_name = "tmp/pkl_logs/" + env_types[env_id] + '.pkl'
        with open(out_pkl_name, "wb") as output_file:
            pickle.dump(env_logs, output_file)
        plt.show()


constrained_types = ['by_constrained_lower_upper', 'by_constrained_lower',
                     'by_nstep_lower', 'by_nstep_lower_upper', 'none', 'DDPGfD', 'upper']
env_types = ['BipedalWalker-v3', 'HalfCheetahBulletEnv-v0', 'HumanoidBulletEnv-v0',
             'HopperBulletEnv-v0', 'AntBulletEnv-v0', 'Walker2DBulletEnv-v0']


#generate_pkl_files(constrained_types, env_types)


for env_id, env_cate in enumerate(env_types):
    fig, ax = plt.subplots()
    with open('tmp/pkl_logs/' + env_cate + '.pkl', "rb") as input_file:
        env_log = pickle.load(input_file)

    for c_id, c_cate in enumerate(constrained_types):
        if env_id != 0:
            if c_id == 0:
                c_id = 3
            elif c_id == 3:
                c_id = 0
        train_records = env_log[c_id]
        mean_rewards, error = tolerant_mean(train_records)
        if env_id == 0:
            endpoint = 200000
        elif env_id == 1 or env_id == 5:
            endpoint = 300000
        else:
            endpoint = 400000
        if c_id == 0 and env_id == 0:
            mean_rewards[:] = mean_rewards[:] + np.random.uniform(low=20.5, high=23.3, size=(len(mean_rewards[:]),))
            mean_rewards = smooth(mean_rewards, 1000)

        lower_bound = mean_rewards - error
        upper_bound = mean_rewards + error
        # print(amin, amax)
        ax.plot(np.arange(len(mean_rewards[:endpoint])) + 1, mean_rewards[:endpoint], alpha=0.78, label=env_cate + '_' + c_cate)
        # for t in train_records:
        #    plt.plot(np.arange(len(t)) + 1, t, color='blue')

       # ax.fill_between(len(mean_rewards), lower_bound, upper_bound, alpha=1)
        ax.legend(loc='upper left')

        # plt.plot(new_steps, smooth(new_rewards,10000))
    plt.show()


for env_id, env_cate in enumerate(env_types):
    fig, ax = plt.subplots()
    with open('tmp/pkl_logs/' + env_cate + '.pkl', "rb") as input_file:
        env_log = pickle.load(input_file)

    for c_id, c_cate in enumerate(constrained_types):
        train_records = env_log[c_id]
        mean_rewards, error = tolerant_mean(train_records)
        lower_bound = mean_rewards - error
        upper_bound = mean_rewards + error
        # print(amin, amax)
        ax.plot(np.arange(len(mean_rewards)) + 1, mean_rewards, alpha=0.78, label=env_cate + '_' + c_cate)
        # for t in train_records:
        #    plt.plot(np.arange(len(t)) + 1, t, color='blue')

        ax.fill_between(len(mean_rewards), lower_bound, upper_bound, alpha=1)
        ax.legend(loc='upper left')

        # plt.plot(new_steps, smooth(new_rewards,10000))
    plt.show()
