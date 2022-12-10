from subprocess import Popen
import os.path as osp
import os


def dw2udw(infile, outfile):
    """
    infile: path to the input file
    outfile: path to the output file
    """
    cwd = os.getcwd()
    current_file_dir = osp.dirname(osp.realpath(__file__))
    os.chdir(current_file_dir)
    Popen(["./bin/utils", "-i", infile, "-o", outfile, "-m", "dw2udw"]).wait()
    os.chdir(cwd)


def udw2dw(infile, outfile):
    """
    infile: path to the input file
    outfile: path to the output file
    """
    cwd = os.getcwd()
    current_file_dir = osp.dirname(osp.realpath(__file__))
    os.chdir(current_file_dir)
    Popen(["./bin/utils", "-i", infile, "-o", outfile, "-m", "udw2dw"]).wait()
    os.chdir(cwd)


def duw2uduw(infile, outfile):
    """
    infile: path to the input file
    outfile: path to the output file
    """
    cwd = os.getcwd()
    current_file_dir = osp.dirname(osp.realpath(__file__))
    os.chdir(current_file_dir)
    Popen(["./bin/utils", "-i", infile, "-o", outfile, "-m", "duw2uduw"]).wait()
    os.chdir(cwd)


def uduw2duw(infile, outfile):
    """
    infile: path to the input file
    outfile: path to the output file
    """
    cwd = os.getcwd()
    current_file_dir = osp.dirname(osp.realpath(__file__))
    os.chdir(current_file_dir)
    Popen(["./bin/utils", "-i", infile, "-o", outfile, "-m", "uduw2duw"]).wait()
    os.chdir(cwd)
