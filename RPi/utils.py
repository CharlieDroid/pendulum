import paramiko
import os
import configparser


class RPIConnect:
    def __init__(self, username="charles", hostname="raspberrypi", password="charlesraspberrypi"):
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=hostname, username=username, password=password)

    def ssh_command(self, command_):
        _, stdout, _ = self.ssh.exec_command(command_)
        return stdout.read().decode()

    def sys_command(self, command_):
        return os.system(command_)


def get_pins():
    config = configparser.ConfigParser()
    config.read("./config.ini")

    button_pin = config.getint("PINS", "button_pin")
    pendulum_pins = config.get("PINS", "pendulum_pins")
    cart_pins = config.get("PINS", "cart_pins")
    motor_pins = config.get("PINS", "motor_pins")

    pendulum_pins = [int(pin) for pin in pendulum_pins.split(",")]
    cart_pins = [int(pin) for pin in cart_pins.split(",")]
    motor_pins = [int(pin) for pin in motor_pins.split(",")]
    return *cart_pins, *pendulum_pins, *motor_pins, button_pin


def get_paths():
    config = configparser.ConfigParser()
    config.read("./config.ini")

    pc_pth = config.get("PATHS", "absolute_path_pc")
    rpi_pth = config.get("PATHS", "absolute_path_rpi")
    return pc_pth, rpi_pth
