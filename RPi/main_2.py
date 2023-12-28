"""
run main in PC
    create actor parameters and send to raspberry pi
    check if there is memory file in pi then delete
    loop:
        wait for memory file in pi
        get memory file and delete it
        learn and then send actor file
        rename memory file to episode_i_data.pkl (optional)
run main in raspberry pi
    load actor file
    waiting for start input
    delete actor file
    loop:
        start training and get episode data
        save episode data
        wait for actor file
        load actor file
        delete actor file
"""
import paramiko

# password = charlesraspberrypi
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("raspberrypi", username="charles")

# get list of files in memory
_, stdout, _ = ssh.exec_command("cd pendulum/memory ; ls -a")
print(stdout.read().decode().split("\n"))

# delete file in memory
_, stdout, _ = ssh.exec_command("cd pendulum ; rm ./memory/main.py")
print(stdout.read().decode().split("\n"))
