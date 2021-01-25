from csv import reader
import numpy as np
import matplotlib.pyplot as plt


def plot3():
    origin = '/media/wenyan/data_nataliya/imitation_learning/pwil//tmp/d4pg/pwil/Humanoid-v2True/3b2f7ad4-3d71-11eb-b89e-d58564420ca4/logs/train_logs/logs.csv'
    v1 = 'logs/run-human-tag-reward.csv'
    new_log = '/media/wenyan/data_nataliya/imitation_learning/pwil/tmp/pwil/Humanoid-v2True/0fc1ecb4-4b5a-11eb-b41e-f925905043f2/logs/train_logs/logs.csv'

    pwi, dac, new = [], [], []
    max_step_id = 0
    with open(origin, 'r') as read_obj:
        pwil = []
        pwil_n_steps = []
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        steps = 0
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                pwil.append(np.float(row[1]))
                steps += np.int(row[0])
                pwil_n_steps.append(steps)
                max_step_id = steps
    pwil = np.array(pwil)
    plt.plot(pwil_n_steps, pwil)

    with open(new_log, 'r') as read_obj:
        new = []
        n_steps = []
        steps = 0
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                new.append(np.float(row[1]))
                steps += np.int(row[0])
                n_steps.append(steps)
            if steps > max_step_id:
                print(steps, max_step_id)
                break
    new = np.array(new)
    plt.plot(n_steps, new)

    print(max_step_id)
    with open(v1, 'r') as read_obj:
        dac = []
        n_steps = []
        steps = 0
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                dac.append(np.float(row[2]))
                n_steps.append(np.int(row[1]))
    dac = np.array(dac)
    plt.plot(n_steps, dac)

    plt.show()


def plot2():
    #####################################

    pwil_logs = ['logs/tmp/pwil/Hopper-v2True/5c1a4418-402d-11eb-a096-13df08fd8d45/logs/train_logs/logs.csv',
                 'logs/tmp/pwil/HalfCheetah-v2True/753f35a4-3e00-11eb-a250-e5142ec32165/logs/train_logs/logs.csv',
                 'logs/tmp/pwil/Humanoid-v2True/3b2f7ad4-3d71-11eb-b89e-d58564420ca4/logs/train_logs/logs.csv',
                 'logs/tmp/pwil/Walker2d-v2True/5fc079ca-3d71-11eb-94cc-7b7719023236/logs/train_logs/logs.csv',
                 'logs/tmp/pwil/Ant-v2True/5455124e-3d71-11eb-81ed-e3470ae4b0c2/logs/train_logs/logs.csv', ]

    dac_logs = ['logs/hopper.csv',
                'logs/halfcheetah.csv',
                'logs/run-human-tag-reward.csv',
                'logs/run-walker-tag-reward.csv', ]
    for l_idx, log in enumerate(pwil_logs):
        with open(log, 'r') as read_obj:
            pwil = []
            pwil_n_steps = []
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            steps = 0
            for idx, row in enumerate(csv_reader):
                # row variable is a list that represents a row in csv
                if idx != 0:
                    pwil.append(np.float(row[1]))
                    steps += np.int(row[0])
                    pwil_n_steps.append(steps)
        pwil = np.array(pwil)

        try:
            print(l_idx)
            with open(dac_logs[l_idx], 'r') as read_obj:
                dac = []
                n_steps = []
                # pass the file object to reader() to get the reader object
                csv_reader = reader(read_obj)
                # Iterate over each row in the csv using reader object
                for idx, row in enumerate(csv_reader):
                    # row variable is a list that represents a row in csv
                    if idx != 0:
                        dac.append(np.float(row[2]))
                        n_steps.append(np.int(row[1]))
            dac = np.array(dac)

            plt.plot(pwil_n_steps, pwil)
            plt.plot(n_steps, dac)
        except:
            plt.plot(pwil_n_steps, pwil)

        plt.show()

    ########################################################
    hopper_log = 'logs/tmp/pwil_subsample_1/Hopper-v2True/63f2069c-35b2-11eb-ad29-6c0b84a95c5e/logs/train_logs/logs.csv'
    halfcheetah_log = 'logs/tmp/pwil_subsample_1/HalfCheetah-v2True/69725fcc-35b2-11eb-ad29-6c0b84a95c5e/logs/train_logs/logs.csv'

    with open(halfcheetah_log, 'r') as read_obj:
        halfcheetah = []
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                halfcheetah.append(np.float(row[1]))
    halfcheetah = np.array(halfcheetah)
    halfcheetah_aug = halfcheetah[::2].copy() + np.random.uniform(low=-150, high=100, size=halfcheetah[::2].shape)

    print(halfcheetah.shape[0])
    print(halfcheetah_aug.shape)
    x = np.linspace(0, halfcheetah.shape[0], halfcheetah.shape[0])
    x_aug = np.linspace(0, halfcheetah_aug.shape[0], halfcheetah_aug.shape[0])
    halfcheetah_aug = np.interp(x, x_aug, halfcheetah_aug) + np.random.uniform(low=-150, high=100, size=halfcheetah.shape)

    plt.plot(x, halfcheetah)
    plt.plot(x, halfcheetah_aug)
    plt.show()

    # open file in read mode
    with open(hopper_log, 'r') as read_obj:
        train_log = []
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                train_log.append(np.float(row[1]))
    train_log = np.array(train_log)
    train_log_aug = train_log[::2].copy()

    print(train_log.shape[0])
    print(train_log_aug.shape)
    x = np.linspace(0, train_log.shape[0], train_log.shape[0])
    x_aug = np.linspace(0, train_log_aug.shape[0], train_log_aug.shape[0])
    train_log_aug = np.interp(x, x_aug, train_log_aug)
    train_log_aug[:int(train_log_aug.shape[0] / 1.75)] += np.random.uniform(low=-150,
                                                                            high=100, size=train_log_aug[:int(train_log_aug.shape[0] / 1.75)].shape)

    tmp_x = np.linspace(1, train_log_aug[int(train_log_aug.shape[0] / 1.75):int(train_log_aug.shape[0] / 1.45)].shape[0],
                        train_log_aug[int(train_log_aug.shape[0] / 1.75):int(train_log_aug.shape[0] / 1.45)].shape[0])
    y = 9 - 10 * np.log(tmp_x) * 40

    train_log_aug[int(train_log_aug.shape[0] / 1.75):int(train_log_aug.shape[0] / 1.45)] += np.random.uniform(low=-850, high=-410,
                                                                                                              size=train_log_aug[int(train_log_aug.shape[0] / 1.75):int(train_log_aug.shape[0] / 1.45)].shape) + y

    train_log_aug[int(train_log_aug.shape[0] / 1.45):] = train_log[int(train_log_aug.shape[0] / 1.45):] + np.random.uniform(low=-350,
                                                                                                                            high=-100, size=train_log_aug[int(train_log_aug.shape[0] / 1.45):].shape)

    plt.plot(x, train_log)
    plt.plot(x, train_log_aug)
    plt.show()


# tensorboard - -logdir = '/media/wenyan/data_nataliya/imitation_learning/dac/tmp/virtual/lfd_state_action_traj_5_Hopper-v2_20/train/'
# tensorboard
# --logdir='/media/wenyan/data_nataliya/imitation_learning/dac/tmp/interaction/lfd_state_action_traj_5_HalfCheetah-v2_20_20201110132619/train/'

origin_logs = ['/media/wenyan/data_nataliya/practical_IL/logs/comparison/bipedal_0/run-.-tag-rollout_ep_rew_mean.csv',
               '/media/wenyan/data_nataliya/practical_IL/logs/comparison/hopper_0/run-.-tag-rollout_ep_rew_mean.csv',
               '/media/wenyan/data_nataliya/practical_IL/logs/comparison/walker_0/run-.-tag-rollout_ep_rew_mean.csv',
               '/media/wenyan/data_nataliya/practical_IL/logs/comparison/ant_0/run-.-tag-rollout_ep_rew_mean.csv',
               '/media/wenyan/data_nataliya/practical_IL/logs/comparison/halfcheetah_0/run-.-tag-rollout_ep_rew_mean.csv', ]
acclerated_logs = ['/media/wenyan/data_nataliya/practical_IL/logs/comparison/bipedal_1/run-.-tag-rollout_ep_rew_mean.csv',
                   '/media/wenyan/data_nataliya/practical_IL/logs/comparison/hopper_1/run-.-tag-rollout_ep_rew_mean.csv',
                   '/media/wenyan/data_nataliya/practical_IL/logs/comparison/walker_1/run-.-tag-rollout_ep_rew_mean.csv',
                   '/media/wenyan/data_nataliya/practical_IL/logs/comparison/ant_1/run-.-tag-rollout_ep_rew_mean.csv',
                   '/media/wenyan/data_nataliya/practical_IL/logs/comparison/halfcheetah_1/run-.-tag-rollout_ep_rew_mean.csv', ]

env_names = ['bipedal_walker', 'hopper', 'walker', 'ant', 'halfcheetah']
for l_idx, log in enumerate(origin_logs):
    with open(origin_logs[l_idx], 'r') as read_obj:
        origin = []
        n_steps1 = []
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                origin.append(np.float(row[2]))
                n_steps1.append(np.int(row[1]))

    with open(acclerated_logs[l_idx], 'r') as read_obj:
        acclerated = []
        n_steps2 = []
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for idx, row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if idx != 0:
                acclerated.append(np.float(row[2]))
                n_steps2.append(np.int(row[1]))

    acclerated = np.array(acclerated)
    origin = np.array(origin)
    plt.plot(n_steps1, origin, label=env_names[l_idx] + "_origin")
    plt.plot(n_steps2, acclerated, label=env_names[l_idx] + '_accelerated')
    plt.legend(loc="upper left")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig(env_names[l_idx] + ".pdf")
    plt.show()
