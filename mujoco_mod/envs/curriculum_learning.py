import numpy as np
from lxml import etree


class Curriculum:
    def __init__(self):
        episodes_to_learn = 300
        self.episodes_to_learn = episodes_to_learn
        self.xml_file_pth = "./mujoco_mod/assets/inverted_double_pendulum.xml"
        self.gravity = -9.80665
        self.gravity_start = -4
        self.curr_gravity = self.gravity_start
        self.a_gravity = np.exp(
            np.log(self.gravity / self.gravity_start) / episodes_to_learn
        )

        self.damping = 0
        self.damping_start = 1.1
        self.curr_damping = self.damping_start
        self.a_damping = -(self.damping - self.damping_start) / episodes_to_learn

        self.damping_cart_start = 0.1
        self.curr_damping_cart = self.damping_cart_start
        self.a_damping_cart = (
            -(self.damping - self.damping_cart_start) / episodes_to_learn
        )

        self.friction_loss = 0
        self.friction_loss_start = 0.6
        self.curr_friction_loss = self.friction_loss_start
        self.a_friction_loss = (
            -(self.friction_loss - self.friction_loss_start) / episodes_to_learn
        )

        self.friction_loss_cart_start = 0.1
        self.curr_friction_loss_cart = self.friction_loss_cart_start
        self.a_friction_loss_cart = (
            -(self.friction_loss - self.friction_loss_cart_start) / episodes_to_learn
        )

        self.curr_steps = 0

        self.element_type = "frictionloss"
        self.start = False

    def curriculum(self):
        if self.start:
            tree = etree.parse(self.xml_file_pth)
            root = tree.getroot()

            root.xpath(".//option")[0].attrib["gravity"] = f"0 0 {self.curr_gravity}"

            self.curr_gravity *= self.a_gravity
            if self.curr_gravity < self.gravity:
                self.curr_gravity = self.gravity

            # damping cart
            root.xpath(".//joint")[1].attrib["damping"] = str(self.curr_damping_cart)
            # damping hinge1
            root.xpath(".//joint")[2].attrib["damping"] = str(self.curr_damping)
            # damping hinge2
            root.xpath(".//joint")[3].attrib["damping"] = str(self.curr_damping)

            self.curr_damping -= self.a_damping
            if self.curr_damping < self.damping:
                self.curr_damping = self.damping

            self.curr_damping_cart -= self.a_damping_cart
            if self.curr_damping_cart < self.damping:
                self.curr_damping_cart = self.damping

            root.xpath(".//joint")[1].attrib["frictionloss"] = str(
                self.curr_friction_loss_cart
            )
            root.xpath(".//joint")[2].attrib["frictionloss"] = str(
                self.curr_friction_loss
            )
            root.xpath(".//joint")[3].attrib["frictionloss"] = str(
                self.curr_friction_loss
            )

            self.curr_friction_loss -= self.a_friction_loss
            if self.curr_friction_loss < self.friction_loss:
                self.curr_friction_loss = self.friction_loss

            self.curr_friction_loss_cart -= self.a_friction_loss_cart
            if self.curr_friction_loss_cart < self.friction_loss:
                self.curr_friction_loss_cart = self.friction_loss

            tree.write(self.xml_file_pth)

            if self.curr_steps % 10 == 0:
                print(f"...curriculum: {self.curr_steps + 1}/{self.episodes_to_learn}")
            self.curr_steps += 1

    def start_environment(self):
        print("...initializing environment parameters curriculum...")
        tree = etree.parse(self.xml_file_pth)
        root = tree.getroot()

        root.xpath(".//option")[0].attrib["gravity"] = f"0 0 {self.gravity_start}"

        self.curr_gravity = self.gravity_start

        # damping cart
        root.xpath(".//joint")[1].attrib["damping"] = str(self.damping_cart_start)
        # damping hinge1
        root.xpath(".//joint")[2].attrib["damping"] = str(self.damping_start)
        # damping hinge2
        root.xpath(".//joint")[3].attrib["damping"] = str(self.damping_start)

        self.curr_damping = self.damping_start
        self.curr_damping_cart = self.damping_cart_start

        root.xpath(".//joint")[1].attrib["frictionloss"] = str(
            self.friction_loss_cart_start
        )
        root.xpath(".//joint")[2].attrib["frictionloss"] = str(self.friction_loss_start)
        root.xpath(".//joint")[3].attrib["frictionloss"] = str(self.friction_loss_start)

        self.curr_friction_loss = self.friction_loss_start
        self.curr_friction_loss_cart = self.friction_loss_cart_start

        tree.write(self.xml_file_pth)
