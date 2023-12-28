import os
import subprocess
import zipfile

print("...running...")
if "rpi.zip" in os.listdir():
    os.remove("./rpi.zip")

files = os.listdir()
with zipfile.ZipFile("rpi.zip", "w") as zipf:
    for file in files:
        if file != ["rpi_push.py", "rpi.zip"]:
            print(file)
            zipf.write(file)
zipf.close()

hostname = "charles@raspberrypi.local"
source = r"C:\Users\Charles\Documents\Python Scripts\Personal\Artificial Intelligence\pendulum\for rpi\rpi.zip"
destination = hostname + ":~/pendulum"
subprocess.call(["scp", source, destination])

subprocess.call(["ssh", hostname, "cd pendulum ;", "unzip -o ~/pendulum/rpi.zip"])
print("...finished...")
