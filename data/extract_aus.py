import glob
import subprocess

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

openface_dir = 'path to OpenFace /build/bin folder'

def openface_command(img_dir, out_dir, name):
    return f'{openface_dir}/FeatureExtraction -fdir {img_dir} -out_dir {out_dir} -of {name} -aus'

print('=============== SAMM DATABASE ===============')
dir = handle_windows_path("path to SAMM dataset")
aus_dir = handle_windows_path("path to SAMM extracted aus")
dirs = []

for name in glob.glob(f'{dir}/**/**'):
    dirs.append(name)

for d in dirs:
    name = d.split('/')[-1]
    command = openface_command(d, aus_dir, name)
    subprocess.run(command, shell=True)
    subprocess.run(f"rm '{aus_dir}/{name.split('.')[0]}_of_details.txt'", shell=True)


print('=============== MMEW DATABASE ===============')
dir = handle_windows_path("path to MMEW dataset")
aus_dir = handle_windows_path("path to MMEW extracted aus")
dirs = []

for name in glob.glob(f'{dir}/**/**'):
    dirs.append(name)

for d in dirs:
    name = d.split('/')[-1]
    command = openface_command(d, aus_dir, name)
    subprocess.run(command, shell=True)
    subprocess.run(f"rm '{aus_dir}/{name.split('.')[0]}_of_details.txt'", shell=True)


print('=============== CASME_II DATABASE ===============')
dir = handle_windows_path("path to CASME II dataset")
aus_dir = handle_windows_path("path to CASME II extracted aus")
dirs = []

for name in glob.glob(f'{dir}/**/**'):
    dirs.append(name)

for d in dirs:
    name = d.split('/')
    name = f"{name[-2]}_{name[-1]}"
    command = openface_command(d, aus_dir, name)
    subprocess.run(command, shell=True)
    subprocess.run(f"rm '{aus_dir}/{name.split('.')[0]}_of_details.txt'", shell=True)