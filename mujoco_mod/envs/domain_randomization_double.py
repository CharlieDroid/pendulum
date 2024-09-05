import numpy as np
from lxml import etree


def map_params_zero_one(x, low, high):
    return (x - low) / (high - low)


def map_params_neg_one_one(x, low, high):
    return (2 * x - low - high) / (high - low)


class DomainRandomization:
    friction_loss_cart = 0.01
    friction_loss_pendulum = 0.0001
    damping = 1
    armature = 0
    length = 0.6
    density_p1 = 1500
    density_p2 = 1000
    gear = 100
    xml_file_pth = "./mujoco_mod/assets/inverted_double_pendulum.xml"

    def __init__(self, highest_score):
        self.highest_score = highest_score
        self.difficulty_level = 0  # [0, 3]
        self.max_action_val = 255  # discretization
        self.pos_sensor_noise_std = 0.005  # 2 / MAX_POS_VALUE
        self.angle_sensor_noise_std = 0.0005  # apply before simp_angle
        self.ep_ = 0
        self.pendulum_balanced = False
        self.disturbance_count = 0
        self.disturbance_range = (2, 5)
        self.disturbance_max_count = np.random.randint(
            low=self.disturbance_range[0], high=self.disturbance_range[1]
        )
        self.disturbance_wait_steps = 0
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
        observation[16] = self.curr_friction_cart
        observation[17] = self.curr_friction_pendulum
        observation[18] = self.curr_damping
        observation[19] = self.curr_armature
        observation[20] = self.curr_gear
        observation[21] = self.curr_length
        observation[22] = self.curr_density_p1
        observation[23] = self.curr_density_p2
        observation[24] = self.curr_force
        observation[25] = self.curr_noise_std
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
        if self.difficulty_level == 0 and score > self.highest_score:
            return self.level_up(ep)
        elif self.difficulty_level == 1 and score > self.highest_score * 0.98:
            return self.level_up(ep)
        elif self.difficulty_level == 2 and score > self.highest_score * 0.94:
            return self.level_up(ep)
        elif self.difficulty_level == 3 and score > self.highest_score * 0.9:
            return self.level_up(ep)
        return False

    def level_up(self, ep):
        self.ep_ = ep
        self.difficulty_level += 1
        print(f"...leveling up to {self.difficulty_level}...")
        return True

    def action(self, action):
        if self.difficulty_level >= 1:
            action = np.trunc(action * self.max_action_val) / self.max_action_val
        return action

    def external_force(self, env, body_idx=3):
        # 3D Force (x, y, z) and 3D torque (theta, phi, psi)
        force = float(np.random.choice([-2, -1, 1, 2]))
        self.curr_force = map_params_neg_one_one(force, -2, 2)
        env.data.xfrc_applied[body_idx] = np.array([0.0, 0.0, 0.0, 0.0, force, 0.0])
        self.disturbance_max_count = np.random.randint(
            low=self.disturbance_range[0], high=self.disturbance_range[1]
        )
        self.disturbance_count += 1
        if self.disturbance_count >= self.disturbance_max_count:
            self.pendulum_balanced = False
            self.disturbance_count = 0
            env.data.xfrc_applied[body_idx] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.curr_force = 0.0

    def observation(self, observation, env):
        disturbance_chance = 0.0
        if abs(observation[6]) < 0.02 and abs(observation[7]) < 0.01:
            if self.difficulty_level >= 2:
                disturbance_chance = 0.3
            elif self.difficulty_level >= 1:
                disturbance_chance = 0.1

        if self.difficulty_level >= 1:
            if np.random.uniform() < disturbance_chance:
                if observation[10] > 0.0:
                    self.external_force(env, body_idx=2)
                self.external_force(env)

        if self.difficulty_level >= 3:
            self.curr_noise_std = 1.0
            pos_noise = np.random.normal(
                loc=0, scale=2 * self.pos_sensor_noise_std, size=1
            )
            angle_noise = np.random.normal(
                loc=0, scale=2 * self.angle_sensor_noise_std, size=1
            )
        elif self.difficulty_level >= 1:
            self.curr_noise_std = 0.5
            pos_noise = np.random.normal(loc=0, scale=self.pos_sensor_noise_std, size=1)
            angle_noise = np.random.normal(
                loc=0, scale=self.angle_sensor_noise_std, size=1
            )

        if self.difficulty_level >= 1:
            a1, a2 = env.data.qpos[1:]
            observation[0] += pos_noise
            observation[1] = np.sin(a1 + angle_noise)
            observation[2] = np.sin(a2 + angle_noise)
            observation[3] = np.cos(a1 + angle_noise)
            observation[4] = np.cos(a2 + angle_noise)
            observation[5] += pos_noise * 2
            observation[6] += angle_noise * 2
            observation[7] += angle_noise * 2

        if self.difficulty_level >= 2:
            self.sensor_store_value([observation[2], observation[4], observation[7]])
            sensor_delay = min(
                int((np.abs(np.random.normal(loc=0.2, scale=3))) / 10),
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
            friction_cart = np.random.uniform(low=0.01, high=0.1)
            root.xpath(".//joint")[1].attrib["frictionloss"] = str(friction_cart)
            self.curr_friction_cart = map_params_zero_one(friction_cart, 0.01, 0.1)

            # friction pendulum 1
            friction_pendulum = np.random.uniform(low=0.0001, high=0.01)
            root.xpath(".//joint")[2].attrib["frictionloss"] = str(friction_pendulum)
            # friction pendulum 2
            root.xpath(".//joint")[3].attrib["frictionloss"] = str(friction_pendulum)
            self.curr_friction_pendulum = map_params_zero_one(
                friction_pendulum, 0.0001, 0.01
            )

            # armature
            if self.difficulty_level >= 4:
                armature = np.random.uniform(low=0, high=0.2) + self.armature
            else:
                armature = np.random.uniform(low=0, high=0.1) + self.armature
            root.xpath(".//joint")[0].attrib["armature"] = str(armature)
            self.curr_armature = map_params_zero_one(armature, 0, 0.2)

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
                length = np.random.uniform(low=-0.15, high=0.1) + self.length
            else:
                ####### NOTE ########
                # !! If changing this, change the ideal bound in td3_fork_double.py and inverted_double_pendulum_mod.py !!
                length = np.random.uniform(low=-0.1, high=0.05) + self.length
            # first pendulum
            root.xpath('.//geom[@name="cpole"]')[0].attrib[
                "fromto"
            ] = f"0 0 0 0.001 0 {length}"
            root.xpath('.//geom[@name="cpole"]')[0].attrib["size"] = f"0.049 {length}"
            root.xpath('.//body[@name="pole2"]')[0].attrib["pos"] = f"0 0 {length}"
            # second pendulum
            root.xpath('.//geom[@name="cpole2"]')[0].attrib[
                "fromto"
            ] = f"0 0 0 0.001 0 {length}"
            root.xpath('.//geom[@name="cpole2"]')[0].attrib["size"] = f"0.049 {length}"
            root.xpath('.//site[@name="tip"]')[0].attrib["pos"] = f"0 0 {length}"
            self.curr_length = map_params_neg_one_one(length, 0.45, 0.75)

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
