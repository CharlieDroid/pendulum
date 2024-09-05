import numpy as np
from lxml import etree


def simp_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class DomainRandomization:
    """
    Level = 0: (Default)
    distubance = 1.0
    damping = 1
    armature = 0
    length = 0.6
    density = 1000
    gear = 100

    Level >= 1:
    disturbance = 0.1
    discretize action to 8 bits
    friction cart = Uniform(low=0.01, high=0.5)
    friction pendulum = Uniform(low=0.01, high=0.0001)
    Gaussian(loc=0,scale=pos_sensor_noise_STD)
    Gaussian(loc=0,scale=angle_sensor_noise_STD)
    armature = Uniform(low=0, high=0.1) + armature

    Level >= 2:
    disturbance = 0.3
    damping = Uniform(low=-1, high=0) + damping
    gear = Uniform(low=-20, high=30) + gear
    (problem is probably armature must stick with 0.1 I think,
    if this is the case then return disturbance to 0.1 and 0.3)

    Level >= 3:
    Gaussian(loc=0,scale=2*pos_sensor_noise_STD)
    Gaussian(loc=0,scale=2*angle_sensor_noise_STD)
    length = Uniform(low=-0.1, high=0.1) + length
    density = Uniform(low=0, high=2000) + density

    Level >= 4:
    length = Uniform(low=-0.15, high=0.15) + length
    density = Uniform(low=-200, high=2500) + density
    gear = Uniform(low=-40, high=50) + gear
    armature = Uniform(low=0, high=0.2) + armature
    disturbance makes it retarded
    """

    def __init__(self):
        self.xml_file_pth = "./mujoco_mod/assets/inverted_pendulum.xml"
        self.difficulty_level = 0  # [0, 3]
        self.max_action_val = 255  # discretization
        self.max_sensor_val = 600
        self.friction_loss_cart = 0  # 0.01 to 0.5 uni
        self.friction_loss_pendulum = 0  # 0.01 to 0.00001 uni
        self.pos_sensor_noise_std = 0.01  # 2 / MAX_POS_VALUE
        self.angle_sensor_noise_std = 0.001  # apply before simp_angle
        self.damping = 1  # -1 to 0 gauss mu=0, sigma=0.1 abs()
        self.armature = 0  # 0 to 0.1 gauss mu=0, sigma=0.01 abs()
        self.length = 0.6  # -1 to 0.5 uni
        self.density = 1000  # 0 to 2000 uni
        self.gear = 100  # -30 to 50 uni
        self.ep_ = 0
        self.pendulum_balanced = False
        self.disturbance_count = 0
        self.disturbance_range = (3, 6)
        self.disturbance_max_count = np.random.randint(
            low=self.disturbance_range[0], high=self.disturbance_range[1]
        )
        self.disturbance_wait_steps = 0

    def check_level_up(self, score, ep):
        if ep - self.ep_ < 10:
            return False
        if self.difficulty_level == 0 and score > 770:
            return self.level_up(ep)
        elif self.difficulty_level == 1 and score > 760:
            return self.level_up(ep)
        elif self.difficulty_level == 2 and score > 720:
            return self.level_up(ep)
        elif self.difficulty_level == 3 and score > 700:
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

    def external_force(self, env):
        # 3D Force (x, y, z) and 3D torque (theta, phi, psi)
        force = float(np.random.choice([-3, -2, 2, 3]))
        env.data.xfrc_applied[2] = np.array([0.0, 0.0, 0.0, 0.0, force, 0.0])
        self.disturbance_max_count = np.random.randint(
            low=self.disturbance_range[0], high=self.disturbance_range[1]
        )
        self.disturbance_count += 1
        if self.disturbance_count >= self.disturbance_max_count:
            self.pendulum_balanced = False
            self.disturbance_count = 0
            env.data.xfrc_applied[2] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def observation(self, observation, env):
        if abs(observation[1]) < 0.07 and abs(observation[3]) < 0.01:
            if self.difficulty_level >= 2:
                disturbance_chance = 0.3
            elif self.difficulty_level >= 1:
                disturbance_chance = 0.1
        else:
            if self.difficulty_level >= 2:
                disturbance_chance = 0.3 / 2
            elif self.difficulty_level >= 1:
                disturbance_chance = 0.1 / 2

        if self.difficulty_level >= 1:
            if np.random.uniform() < disturbance_chance:
                self.external_force(env)

        if self.difficulty_level >= 3:
            pos_noise = np.random.normal(
                loc=0, scale=2 * self.pos_sensor_noise_std, size=1
            )
            angle_noise = np.random.normal(
                loc=0, scale=2 * self.angle_sensor_noise_std, size=1
            )
        elif self.difficulty_level >= 1:
            pos_noise = np.random.normal(loc=0, scale=self.pos_sensor_noise_std, size=1)
            angle_noise = np.random.normal(
                loc=0, scale=self.angle_sensor_noise_std, size=1
            )

        if self.difficulty_level >= 1:
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
                np.random.uniform(low=0.01, high=0.1) + self.friction_loss_cart
            )
            # friction pendulum
            root.xpath(".//joint")[2].attrib["frictionloss"] = str(
                np.random.uniform(low=0.0001, high=0.01) + self.friction_loss_pendulum
            )

            # armature
            if self.difficulty_level >= 4:
                root.xpath(".//joint")[0].attrib["armature"] = str(
                    np.random.uniform(low=0, high=0.2) + self.armature
                )
            else:
                root.xpath(".//joint")[0].attrib["armature"] = str(
                    np.random.uniform(low=0, high=0.1) + self.armature
                )

        if self.difficulty_level >= 2:
            # damping
            root.xpath(".//joint")[0].attrib["damping"] = str(
                np.random.uniform(low=-1, high=0) + self.damping
            )

            # gear
            if self.difficulty_level >= 4:
                gear = str(int(np.random.uniform(low=-40, high=50) + self.gear))
            else:
                gear = str(int(np.random.uniform(low=-20, high=30) + self.gear))
            root.xpath(".//actuator")[0].find('.//motor[@name="slide"]').attrib[
                "gear"
            ] = gear

        if self.difficulty_level >= 3:
            # pendulum length
            if self.difficulty_level >= 4:
                length = np.random.uniform(low=-0.15, high=0.15) + self.length
            else:
                length = np.random.uniform(low=-0.1, high=0.1) + self.length
            root.xpath('.//geom[@name="cpole"]')[0].attrib[
                "fromto"
            ] = f"0 0 0 0.001 0 {length}"
            root.xpath('.//geom[@name="cpole"]')[0].attrib["size"] = f"0.049 {length}"

            # pendulum density
            if self.difficulty_level >= 4:
                root.xpath('.//geom[@name="cpole"]')[0].attrib["density"] = str(
                    np.random.uniform(low=-200, high=2500) + self.density
                )
            else:
                root.xpath('.//geom[@name="cpole"]')[0].attrib["density"] = str(
                    np.random.uniform(low=0, high=2000) + self.density
                )

        if tree:
            tree.write(self.xml_file_pth)

    def reset_environment(self):
        print("...resetting environment parameters domain randomization...")
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

        root.xpath(".//option")[0].attrib["gravity"] = f"0 0 -9.80665"

        tree.write(self.xml_file_pth)
