import os
import zipfile

from utils import get_paths, RPIConnect


if __name__ == "__main__":
    print("...running...")
    send_python_files = False
    send_cpp_files = True
    if send_python_files:
        blacklist = (
            "rpi_push.py",
            "rpi.zip",
            "__pycache__",
            "td3_fork_buffer.pkl",
            "buffer_warmup.pkl",
            "td3_fork.chkpt",
            "td3_fork_good.chkpt",
            "td3_fork_sim.chkpt",
            "td3_fork_sim_16x16.chkpt",
            "td3_fork_sim_32x32.chkpt",
            "td3_fork_sim_25Hz.chkpt",
            "runs",
            "backup_warmup_data",
            "cmake-build-debug",
            "CMakeLists.txt",
        )
        if "rpi.zip" in os.listdir():
            os.remove("./rpi.zip")

        # this can be improved using os.walk() but am too lazy and this only checks up to 2 directories deep
        files = os.listdir()
        with zipfile.ZipFile("rpi.zip", "w") as zipf:
            for file in files:
                if file not in blacklist:
                    if "." not in file:
                        zipf.write(file)
                        print(file)
                        files_ = os.listdir(file)
                        for file_ in files_:
                            if file_ not in blacklist:
                                zipf.write(f"./{file}/{file_}")
                                print(file_)
                    else:
                        zipf.write(file)
                        print(file)
        zipf.close()

        # if error then find config file and change it there for now
        hostname = "charles@raspberrypi"
        pc_pth, pi_pth = get_paths()
        source = os.path.join(pc_pth, "rpi.zip").replace("\\", "/")
        destination = hostname + f":{pi_pth}"
        print("...sending...")
        rpi = RPIConnect()
        rpi.sys_command(f'scp "{source}" {destination}')
        rpi.ssh_command(f"cd pendulum ; unzip -o {pi_pth}/rpi.zip")
        print("...finished...")
        os.remove("./rpi.zip")
        print("...removed rpi.zip...")
    elif send_cpp_files:
        blacklist = (
            "rpi_push.py",
            "rpi.zip",
            "__pycache__",
            "cmake-build-debug",
            "CMakeLists.txt",
            "libs",
            ".idea",
            "charles_notes.txt",
        )
        os.chdir("../RPi_CPP")
        pc_pth, pi_pth = get_paths()
        if "rpi.zip" in os.listdir():
            os.remove("./rpi.zip")

        # this can be improved using os.walk() but am too lazy and this only checks up to 2 directories deep
        files = os.listdir()
        with zipfile.ZipFile("rpi.zip", "w") as zipf:
            for file in files:
                if file not in blacklist:
                    if "." not in file:
                        zipf.write(file)
                        print(file)
                        files_ = os.listdir(file)
                        for file_ in files_:
                            if file_ not in blacklist:
                                zipf.write(f"./{file}/{file_}")
                                print(file_)
                    else:
                        zipf.write(file)
                        print(file)
        zipf.close()

        # if error then find config file and change it there for now
        hostname = "charles@raspberrypi"
        # replace pc path to the rpi cpp folder
        # pc_pth = os.path.join(os.path.dirname(os.getcwd()), "RPi_CPP").replace(
        #     "\\", "/"
        # )
        source = os.path.join(pc_pth, "rpi.zip").replace("\\", "/")
        destination = hostname + f":{pi_pth}"
        print("...sending...")
        trials = 0
        while True:
            rpi = RPIConnect()
            rpi.sys_command(f'scp "{source}" {destination}')
            if "rpi.zip" in rpi.ssh_command("cd pendulum ; ls -a").split("\n"):
                print("...sent...")
                break
            trials += 1
            print(f"Trial #{trials}")
            if trials > 20:
                import sys

                sys.exit("...failed to send...")

        rpi.ssh_command(
            f"cd pendulum ; unzip -o {pi_pth}/rpi.zip"
        )  # deleting rpi.zip apparently fixes connection probs
        os.remove("rpi.zip")
        rpi.ssh_command(f"cd pendulum ; rm {pi_pth}/rpi.zip")
        print("...removed rpi.zip...")
