# Adapted from https://github.com/Pletron/LFWgender/blob/master/name_to_gender.py
# "This is a python script that calls the genderize.io API with the first name of the person in the image.
# If the confidence is more than 90% the file gets copied to the corresponding gender folder: male/female.
# The image is ignored if the name is not found in the genderize.io database."

import os
import sys
from os.path import join, exists
from django.http import request
from os import rename
import requests
import shutil

__author__ = 'Philip Masek'

def main(argv):
    fileList = []
    fileSize = 0
    folderCount = 0
    rootdir = "./lfw"
    maleFolder = "./result/male"
    femaleFolder = "./result/female"
    count = 0
    tmp = ""

    for root, subFolders, files in os.walk(rootdir):
        for file in files:
            f = os.path.join(root, file)
            fileSize = fileSize + os.path.getsize(f)
            fileSplit = file.split("_")
            fileList.append(f)
            count += 1

            file_exists = exists("%s/%s" % (maleFolder, file)) or exists("%s/%s" % (maleFolder, file))

            if (not file_exists):

                if count == 1:
                    result = requests.get("https://api.genderize.io/?name=%s" % fileSplit[0])
                    try:
                        result = result.json()
                    except:
                        break;
                    tmp = fileSplit[0]
                elif tmp != fileSplit[0]:
                    result = requests.get("https://api.genderize.io/?name=%s" % fileSplit[0])
                    try:
                        result = result.json()
                    except:
                        break;
                    tmp = fileSplit[0]
                else:
                    tmp = fileSplit[0]

                try:
                    print
                    result;
                    if float(result['probability']) > 0.9:
                        if result['gender'] == 'male':
                            shutil.copyfile(f, "%s/%s" % (maleFolder, file))
                        elif result['gender'] == 'female':
                            shutil.copyfile(f, "%s/%s" % (femaleFolder, file))
                except Exception as e:
                    break;


if __name__ == "__main__":
    main(sys.argv)
