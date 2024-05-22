import os
import zipfile

cwd = os.getcwd()
archive = zipfile.ZipFile(cwd + "/gym/PPO_tutorial.zip", "r")
for f in archive.filelist:
    print(f.filename)