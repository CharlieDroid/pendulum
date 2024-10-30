import numpy as np
from lxml import etree


def map_params_zero_one(x, low, high):
    return (x - low) / (high - low)


def map_params_neg_one_one(x, low, high):
    return (2 * x - low - high) / (high - low)


class DomainRandomization:
    """
    Level = 0: (Default)
    action_noise = 0.1
    damping = 1
    armature = 0
    length_first_pendulum = 0.3
    length_second_pendulum = 0.33
    density_first_pendulum = 1500  (heavier since it will hold an encoder)
    density_second_pendulum = 1000
    gear = 100

    Level >= 1:
    action_noise = 0.05
    disturbance = 1%
    friction cart = 0.001 to 0.01
    friction pendulum = 0.001 to 0.01
    pos and sensor noise discretization and gaussian noise of 0.25
    armature = 0 to 0.03

    Level >= 2:
    damping = 0 to 1
    gear = 80 to 120
    sensor_delays = true

    Level >= 3:
    action_noise = 0.0
    pos and sensor noise * 2
    disturbance = 2%
    length_first_pendulum = 0.24 to 0.33
    length_second_pendulum = 0.27 to 0.36
    density_first_pendulum = 1500 to 1900
    density_second_pendulum = 1000 to 1200

    Level >= 4:
    action_noise = 0.0
    length_first_pendulum = 0.19 to 0.36
    length_second_pendulum = 0.22 to 0.39
    density_first_pendulum = 1200 to 2000
    density_second_pendulum = 900 to 1300
    gear = 75 to 135
    armature = 0 to 0.06
    """

    friction_loss_cart = 0.01
    friction_loss_pendulum = 0.0001
    damping = 1
    armature = 0
    length = 0.3
    density_p1 = 1500
    density_p2 = 1000
    gear = 100
    xml_file_pth = "./mujoco_mod/assets/inverted_double_pendulum.xml"

    def __init__(self, highest_score):
        self.highest_score = highest_score
        self.difficulty_level = 0  # [0, 3]
        # no need discretization since in giga, it is 14 bits
        self.encoder_sensor_noise_std = 1  # 1-3 steps is the noise of usual encoders
        self.angle_to_steps = 300 / np.pi  # 600 steps per 360 degrees
        self.steps_to_angle = np.pi / 300
        self.angle_velo_to_steps = (300 * 0.01) / np.pi
        self.steps_to_angle_velo = np.pi / (300 * 0.01)
        self.pos_velo_to_steps = (13491 * 0.01) / 2
        self.steps_to_pos_velo = 2 / (13491 * 0.01)
        self.sensor_noise_std = 0.001
        self.sensor_velo_noise_std = 0.01
        self.ep_ = 0
        self.pendulum_balanced = False
        self.disturbance_count = 0
        self.disturbance_range = (2, 5)
        self.disturbance_max_count = np.random.randint(
            low=self.disturbance_range[0], high=self.disturbance_range[1]
        )
        self.disturbance_on = False
        self.disturbance_wait_steps = 0
        self.max_disturbance = 1.0
        self.disturbance_body_idx = 3  # idx = 3 is the 2nd pendulum, idx = 2 is the 1st pendulum
        self.sensor_max_delay = 4
        self.sensor_data = [0 for _ in range(self.sensor_max_delay + 1)]
        self.sensor_stored = False

        self.curr_friction_cart = 0.0
        self.curr_friction_pendulum = 0.0
        self.curr_damping = 0.0
        self.curr_armature = 0.0
        self.curr_gear = 0.0
        self.curr_length = 0.0
        self.curr_density_p1 = 0.0
        self.curr_density_p2 = 0.0
        self.curr_force = 0.0
        self.curr_noise_std = 0.0

    def add_system_params(self, observation):
        observation[12] = self.curr_friction_cart
        observation[13] = self.curr_friction_pendulum
        observation[14] = self.curr_damping
        observation[15] = self.curr_armature
        observation[16] = self.curr_gear
        observation[17] = self.curr_length
        observation[18] = self.curr_density_p1
        observation[19] = self.curr_density_p2
        observation[20] = self.curr_force
        observation[21] = self.curr_noise_std
        return observation

    def sensor_store_value(self, val):
        if not self.sensor_stored:
            self.sensor_stored = True
            for i in range(len(self.sensor_data)):
                self.sensor_data[i] = val
        else:
            self.sensor_data[1:] = self.sensor_data[:-1]
            self.sensor_data[0] = val

    def sensor_get_value(self, sensor_delay):
        return self.sensor_data[sensor_delay]

    def check_level_up(self, score, ep):
        if ep - self.ep_ < 10:
            return False
        if self.difficulty_level == 0 and score > self.highest_score * 0.85:
            return self.level_up(ep)
        elif self.difficulty_level == 1 and score > self.highest_score * 0.78:
            return self.level_up(ep)
        elif self.difficulty_level == 2 and score > self.highest_score * 0.74:
            return self.level_up(ep)
        elif self.difficulty_level == 3 and score > self.highest_score * 0.56:
            return self.level_up(ep)
        return False

    def level_up(self, ep):
        self.ep_ = ep
        self.difficulty_level += 1
        print(f"...leveling up to {self.difficulty_level}...")
        return True

    def action(self, action):
        noise = 0.0
        if self.difficulty_level >= 3:
            noise = 0.0
            # if np.random.uniform() < 0.05 and not self.disturbance_on:  # disturbance at 20% of the time
            #     action[0] = float(np.random.choice((-self.max_disturbance, self.max_disturbance)))
        elif self.difficulty_level >= 1:
            noise = np.random.normal(0, 0.05)
            # if np.random.uniform() < 0.025 and not self.disturbance_on:
            #     action[0] = float(np.random.choice((-self.max_disturbance, self.max_disturbance)))
        elif self.difficulty_level == 0:
            noise = np.random.normal(0, 0.1)

        action = np.clip(action + noise, -1.0, 1.0)
        return action

    def external_force(self, env, easy=True):
        # 3D Force (x, y, z) and 3D torque (theta, phi, psi)
        hard_force_list = [-8, -6, -4, -2, 2, 4, 6, 8]
        if easy:
            force = float(np.random.choice([-4, -3, -2, 2, 3, 4]))
        else:
            force = float(np.random.choice(hard_force_list))
        # to be used for critic and systems network
        self.curr_force = map_params_neg_one_one(force, hard_force_list[0], hard_force_list[-1])

        # I want to hit 2nd pendulum more
        self.disturbance_body_idx = int(np.random.choice([2, 3, 3]))
        env.data.xfrc_applied[self.disturbance_body_idx] = np.array([0.0, 0.0, 0.0, 0.0, force, 0.0])
        self.disturbance_max_count = np.random.randint(
            low=self.disturbance_range[0], high=self.disturbance_range[1]
        )
        self.disturbance_on = True

    def observation(self, observation, env):
        disturbance_chance = 0.0
        easy = True
        # if a1 vel and a2 vel are close to zero then it has balanced
        if abs(observation[6]) < 0.02 and abs(observation[7]) < 0.01:
            if self.difficulty_level >= 2 and not self.disturbance_on:
                disturbance_chance = 0.002
                easy = False
            elif self.difficulty_level >= 1 and not self.disturbance_on:
                disturbance_chance = 0.001
                easy = False

        if self.difficulty_level >= 1:
            if (np.random.uniform() < disturbance_chance) and not self.disturbance_on:
                self.external_force(env, easy=easy)
            elif self.disturbance_on:
                self.disturbance_count += 1
                if self.disturbance_count >= self.disturbance_max_count:
                    self.disturbance_count = 0
                    env.data.xfrc_applied[self.disturbance_body_idx] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self.disturbance_on = False

        if self.difficulty_level >= 3:
            self.curr_noise_std = 1.0
            pos_noise_step = int(np.random.normal(loc=0, scale=self.encoder_sensor_noise_std * 2, size=1)[0])

            a1, a2 = env.data.qpos[1:]
            angle1_noise_step = int(np.random.normal(loc=0, scale=self.encoder_sensor_noise_std * 2, size=1)[0])
            angle2_noise_step = int(np.random.normal(loc=0, scale=self.encoder_sensor_noise_std * 2, size=1)[0])

            velo_noise = np.random.normal(loc=0, scale=self.sensor_velo_noise_std, size=1)[0]
            pos_and_angle_noise = np.random.normal(loc=0, scale=self.sensor_noise_std, size=1)[0]

        elif self.difficulty_level >= 1:
            self.curr_noise_std = 0.5

            pos_noise_step = int(np.random.normal(loc=0, scale=self.encoder_sensor_noise_std, size=1)[0])

            a1, a2 = env.data.qpos[1:]
            angle1_noise_step = int(np.random.normal(loc=0, scale=self.encoder_sensor_noise_std, size=1)[0])
            angle2_noise_step = int(np.random.normal(loc=0, scale=self.encoder_sensor_noise_std, size=1)[0])

            velo_noise = np.random.normal(loc=0, scale=self.sensor_velo_noise_std, size=1)[0]
            pos_and_angle_noise = np.random.normal(loc=0, scale=self.sensor_noise_std, size=1)[0]


        if self.difficulty_level >= 1:
            # discretize then add step noise then convert to float again
            pos_new = float((int((observation[0] + 1) * 6745.5) + pos_noise_step) * 2 / 13491 - 1)
            pos_velo_new = float(
                (int(observation[5] * self.pos_velo_to_steps) + pos_noise_step) * self.steps_to_pos_velo)

            a1_new = float((int(a1 * self.angle_to_steps) + angle1_noise_step) * self.steps_to_angle)
            a1_velo_new = float((int(
                observation[6] * self.angle_velo_to_steps * 30) + angle1_noise_step) * self.steps_to_angle_velo / 30)

            a2_new = float((int(a2 * self.angle_to_steps) + angle2_noise_step) * self.steps_to_angle)
            a2_velo_new = float((int(
                observation[7] * self.angle_velo_to_steps * 30) + angle2_noise_step) * self.steps_to_angle_velo / 30)
            observation[0] = pos_new
            observation[1] = np.sin(a1_new)
            observation[2] = np.sin(a2_new)
            observation[3] = np.cos(a1_new)
            observation[4] = np.cos(a2_new)
            observation[5] = pos_velo_new
            observation[6] = a1_velo_new
            observation[7] = a2_velo_new

        if self.difficulty_level >= 2:
            self.sensor_store_value([observation[2], observation[4], observation[7]])
            sensor_delay = min(
                int((np.abs(np.random.normal(loc=0, scale=3.8))) / 10),
                self.sensor_max_delay,
            )
            if sensor_delay > 0:
                observation[2], observation[4], observation[7] = self.sensor_get_value(
                    sensor_delay
                )
        return self.add_system_params(observation)

    def environment(self):
        tree = None
        if self.difficulty_level >= 1:
            tree = etree.parse(self.xml_file_pth)
            root = tree.getroot()

            # randomize environment
            # friction cart
            friction_cart = np.random.uniform(low=0.001, high=0.01)
            root.xpath(".//joint")[1].attrib["frictionloss"] = str(friction_cart)
            self.curr_friction_cart = map_params_zero_one(friction_cart, 0.001, 0.01)

            # friction pendulum 1
            friction_pendulum = np.random.uniform(low=0.0001, high=0.001)
            root.xpath(".//joint")[2].attrib["frictionloss"] = str(friction_pendulum)
            # friction pendulum 2
            root.xpath(".//joint")[3].attrib["frictionloss"] = str(friction_pendulum)
            self.curr_friction_pendulum = map_params_zero_one(
                friction_pendulum, 0.0001, 0.001
            )

            # armature
            if self.difficulty_level >= 4:
                armature = np.random.uniform(low=0, high=0.06) + self.armature
            else:
                armature = np.random.uniform(low=0, high=0.03) + self.armature
            root.xpath(".//joint")[0].attrib["armature"] = str(armature)
            self.curr_armature = map_params_zero_one(armature, 0, 0.06)

        if self.difficulty_level >= 2:
            # damping
            damping = np.random.uniform(low=-1, high=0) + self.damping
            root.xpath(".//joint")[0].attrib["damping"] = str(damping)
            self.curr_damping = map_params_zero_one(damping, 1, 0)

            # gear
            if self.difficulty_level >= 4:
                gear = int(np.random.uniform(low=-30, high=40) + self.gear)
            else:
                gear = int(np.random.uniform(low=-15, high=25) + self.gear)
            root.xpath(".//actuator")[0].find('.//motor[@name="slide"]').attrib[
                "gear"
            ] = str(gear)
            self.curr_gear = map_params_neg_one_one(gear, 60, 140)

        if self.difficulty_level >= 3:
            # pendulum length
            if self.difficulty_level >= 4:
                ####### NOTE ########
                # !! If changing this, change the ideal bound in td3_fork_double.py and inverted_double_pendulum_mod.py !!
                length = np.random.uniform(low=-0.11, high=0.06) + self.length
            else:
                ####### NOTE ########
                # !! If changing this, change the ideal bound in td3_fork_double.py and inverted_double_pendulum_mod.py !!
                length = np.random.uniform(low=-0.06, high=0.03) + self.length
            # I have decided that the second pendulum be a bit longer since the first pendulum is heavier
            length_constant = 0.03  # 30mm
            # first pendulum
            root.xpath('.//geom[@name="cpole"]')[0].attrib[
                "fromto"
            ] = f"0 0 0 0.001 0 {length}"
            root.xpath('.//geom[@name="cpole"]')[0].attrib["size"] = f"0.049 {length}"
            root.xpath('.//body[@name="pole2"]')[0].attrib["pos"] = f"0 0 {length}"
            # second pendulum
            root.xpath('.//geom[@name="cpole2"]')[0].attrib[
                "fromto"
            ] = f"0 0 0 0.001 0 {length + length_constant}"
            root.xpath('.//geom[@name="cpole2"]')[0].attrib["size"] = f"0.049 {length + length_constant}"
            root.xpath('.//site[@name="tip"]')[0].attrib["pos"] = f"0 0 {length + length_constant}"
            # length constant wasn't added here since it is a constant
            self.curr_length = map_params_neg_one_one(length, 0.19, 0.41)

            # pendulum density
            if self.difficulty_level >= 4:
                density_p1 = np.random.uniform(low=-100, high=2200) + self.density_p1
                density_p2 = np.random.uniform(low=-100, high=2200) + self.density_p2
            else:
                density_p1 = np.random.uniform(low=0, high=1800) + self.density_p1
                density_p2 = np.random.uniform(low=0, high=1800) + self.density_p2
            root.xpath('.//geom[@name="cpole"]')[0].attrib["density"] = str(density_p1)
            root.xpath('.//geom[@name="cpole2"]')[0].attrib["density"] = str(density_p2)
            self.curr_density_p1 = map_params_neg_one_one(density_p1, -700, 3700)
            self.curr_density_p2 = map_params_neg_one_one(density_p2, -1200, 3200)

        if tree:
            tree.write(self.xml_file_pth)

    def reset_environment(self):
        print("...resetting environment parameters domain randomization...")
        self.sensor_data = [0 for _ in range(self.sensor_max_delay + 1)]
        self.sensor_stored = False

        tree = etree.parse(self.xml_file_pth)
        root = tree.getroot()

        # randomize environment
        # friction cart
        root.xpath(".//joint")[1].attrib["frictionloss"] = str(self.friction_loss_cart)
        # friction pendulum 1
        root.xpath(".//joint")[2].attrib["frictionloss"] = str(
            self.friction_loss_pendulum
        )
        # friction pendulum 2
        root.xpath(".//joint")[3].attrib["frictionloss"] = str(
            self.friction_loss_pendulum
        )

        root.xpath(".//actuator")[0].find('.//motor[@name="slide"]').attrib[
            "gear"
        ] = str(self.gear)
        root.xpath(".//joint")[0].attrib["damping"] = str(self.damping)
        root.xpath(".//joint")[0].attrib["armature"] = str(self.armature)

        length = self.length
        root.xpath('.//geom[@name="cpole"]')[0].attrib[
            "fromto"
        ] = f"0 0 0 0.001 0 {length}"
        root.xpath('.//geom[@name="cpole"]')[0].attrib["size"] = f"0.049 {length}"
        root.xpath('.//body[@name="pole2"]')[0].attrib["pos"] = f"0 0 {length}"
        # pendulum 1 density
        root.xpath('.//geom[@name="cpole"]')[0].attrib["density"] = str(self.density_p1)

        root.xpath('.//geom[@name="cpole2"]')[0].attrib[
            "fromto"
        ] = f"0 0 0 0.001 0 {length}"
        root.xpath('.//geom[@name="cpole2"]')[0].attrib["size"] = f"0.049 {length}"
        root.xpath('.//site[@name="tip"]')[0].attrib["pos"] = f"0 0 {length}"
        # pendulum 2 density
        root.xpath('.//geom[@name="cpole2"]')[0].attrib["density"] = str(
            self.density_p2
        )

        root.xpath(".//option")[0].attrib["gravity"] = f"0 0 -9.80665"

        tree.write(self.xml_file_pth)
