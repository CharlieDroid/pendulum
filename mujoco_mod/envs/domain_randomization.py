import numpy as np
from lxml import etree


def simp_angle(a):
    _2pi = 2 * np.pi
    if a > np.pi:
        return simp_angle(a - _2pi)
    elif a < -np.pi:
        return simp_angle(a + _2pi)
    else:
        return a


class DomainRandomization:
    def __init__(self):
        self.xml_file_pth = "./mujoco_mod/assets/inverted_pendulum.xml"
        self.difficulty_level = 0  # 0 to 3 (including)
        self.max_disturbance = 1.0  # -1 to 1
        self.max_action_val = 255  # discretization
        self.max_sensor_val = 600
        self.friction_loss_cart = 0  # 0.01 to 0.5 uni
        self.friction_loss_pendulum = 0  # 0.01 to 0.00001 uni
        self.pos_sensor_noise_std = 2 / 13491  # 2 / MAX_POS_VALUE
        self.angle_sensor_noise_std = np.pi / 299.5  # apply before simp_angle
        self.damping = 1  # -1 to 0 gauss mu=0, sigma=0.1 abs()
        self.armature = 0  # 0 to 0.1 gauss mu=0, sigma=0.01 abs()
        self.length = 0.6  # -1 to 0.5 uni
        self.density = 1000  # 0 to 2000 uni
        self.gear = 100  # -30 to 50 uni
        self.ep_ = 0

    def check_level_up(self, score, ep):
        if ep - self.ep_ < 10:
            return False
        if self.difficulty_level == 0 and score > 770:
            return self.level_up(ep)
        elif self.difficulty_level == 1 and score > 760:
            return self.level_up(ep)
        elif self.difficulty_level == 2 and score > 700:
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
        if self.difficulty_level >= 2:
            if np.random.uniform() < 0.3:  # disturbance at 30% of the time
                if np.random.uniform() < 0.5:
                    action += self.max_disturbance
                else:
                    action -= self.max_disturbance
        elif self.difficulty_level >= 1:
            if np.random.uniform() < 0.1:  # disturbance at 10% of the time
                if np.random.uniform() < 0.5:
                    action += self.max_disturbance
                else:
                    action -= self.max_disturbance
        return action

    def observation(self, observation):
        if self.difficulty_level >= 1:
            observation[0] = np.trunc(observation[0] * 13491) / 13491
            observation[1] = np.trunc(observation[1] * 600) / 600

        if self.difficulty_level >= 3:
            pos_noise = np.random.normal(
                loc=0, scale=2 * self.pos_sensor_noise_std, size=1
            )
            angle_noise = np.random.normal(
                loc=0, scale=2 * self.angle_sensor_noise_std, size=1
            )
        elif self.difficulty_level >= 2:
            pos_noise = np.random.normal(loc=0, scale=self.pos_sensor_noise_std, size=1)
            angle_noise = np.random.normal(
                loc=0, scale=self.angle_sensor_noise_std, size=1
            )

        if self.difficulty_level >= 2:
            observation[0] += pos_noise
            observation[1] += angle_noise
            observation[2] += pos_noise * 2
            observation[3] += angle_noise * 2
        return observation

    def environment(self):
        tree = None
        if self.difficulty_level >= 1:
            tree = etree.parse(self.xml_file_pth)
            root = tree.getroot()

            # randomize environment
            # friction cart
            root.xpath(".//joint")[1].attrib["frictionloss"] = str(
                np.random.uniform(low=0.01, high=0.5) + self.friction_loss_cart
            )
            # friction pendulum
            root.xpath(".//joint")[2].attrib["frictionloss"] = str(
                np.random.uniform(low=0.01, high=0.00001) + self.friction_loss_pendulum
            )

        if self.difficulty_level >= 2:
            root.xpath(".//actuator")[0].find('.//motor[@name="slide"]').attrib[
                "gear"
            ] = str(int(np.random.uniform(low=-20, high=30) + self.gear))
            root.xpath(".//joint")[0].attrib["damping"] = str(
                np.random.uniform(low=-1, high=0) + self.damping
            )
            root.xpath(".//joint")[0].attrib["armature"] = str(
                np.random.uniform(low=0, high=0.2) + self.armature
            )

        if self.difficulty_level >= 3:
            # pendulum length
            length = np.random.uniform(low=-0.1, high=0.1) + self.length
            root.xpath('.//geom[@name="cpole"]')[0].attrib[
                "fromto"
            ] = f"0 0 0 0.001 0 {length}"
            root.xpath('.//geom[@name="cpole"]')[0].attrib["size"] = f"0.049 {length}"
            # pendulum density
            root.xpath('.//geom[@name="cpole"]')[0].attrib["density"] = str(
                np.random.uniform(low=0, high=2000) + self.density
            )

        if tree:
            tree.write(self.xml_file_pth)

    def reset_environment(self):
        print("...resetting environment parameters...")
        tree = etree.parse(self.xml_file_pth)
        root = tree.getroot()

        # randomize environment
        # friction cart
        root.xpath(".//joint")[1].attrib["frictionloss"] = str(self.friction_loss_cart)
        # friction pendulum
        root.xpath(".//joint")[2].attrib["frictionloss"] = str(
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
        # pendulum density
        root.xpath('.//geom[@name="cpole"]')[0].attrib["density"] = str(self.density)

        tree.write(self.xml_file_pth)
