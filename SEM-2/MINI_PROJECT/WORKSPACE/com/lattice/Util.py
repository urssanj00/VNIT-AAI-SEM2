# import OS module
import os


class Util:
    @staticmethod
    def list_files_of_a_dir(dir_path):
        # Get the list of all files and directories
        dir_list = os.listdir(dir_path)
        print("Files and directories in '", dir_path, "' :")
        # prints all files
        print(dir_list)
        return dir_list


