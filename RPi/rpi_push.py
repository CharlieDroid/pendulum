import os
import paramiko
import zipfile

from utils import get_paths, RPIConnect


if __name__ == "__main__":
    print("...running...")
    if "rpi.zip" in os.listdir():
        os.remove("./rpi.zip")

    files = os.listdir()
    with zipfile.ZipFile("rpi.zip", "w") as zipf:
        for file in files:
            if file not in ["rpi_push.py", "rpi.zip", "__pycache__"]:
                if "." not in file:
                    zipf.write(file)
                    print(file)
                    files_ = os.listdir(file)
                    for file_ in files_:
                        if file_ not in ["rpi_push.py", "rpi.zip", "__pycache__"]:
                            zipf.write(f"./{file}/{file_}")
                            print(file_)
                else:
                    zipf.write(file)
                    print(file)
    zipf.close()

    # if error then find config file and change it there for now
    hostname = "charles@raspberrypi"
    pc_pth, pi_pth = get_paths()
    source = os.path.join(pc_pth, "rpi.zip")
    destination = hostname + f":{pi_pth}"
    print("...sending...")
    rpi = RPIConnect()
    rpi.sys_command(f"scp {source} {destination}")
    rpi.ssh_command(f"cd pendulum ; unzip -o {pi_pth}/rpi.zip")
    print("...finished...")
