import os.path as osp
import os
import sys
from subprocess import Popen
import myLogger

gapbs_dir = osp.join(osp.dirname(osp.abspath(__file__)), "gapbs")
current_file_dir = osp.dirname(osp.abspath(__file__))


def bfs(**kwargs):
    bin_path = osp.join(gapbs_dir, "bfs")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/bfs does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def bc(**kwargs):
    bin_path = osp.join(gapbs_dir, "bc")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/bc does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-i', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def cc(**kwargs):
    bin_path = osp.join(gapbs_dir, "cc")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/cc does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def cc_sv(**kwargs):
    bin_path = osp.join(gapbs_dir, "cc_sv")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/cc_sv does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def pr(**kwargs):
    bin_path = osp.join(gapbs_dir, "pr")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/pr does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-i', '-t', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def pr_spmv(**kwargs):
    bin_path = osp.join(gapbs_dir, "pr_spmv")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/pr_spmv does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-i', '-t', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def sssp(**kwargs):
    bin_path = osp.join(gapbs_dir, "sssp")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/sssp does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-d', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)


def tc(**kwargs):
    bin_path = osp.join(gapbs_dir, "tc")
    cwd = os.getcwd()
    assert os.path.exists(bin_path), "gapbs/tc does not exist. Run `make` in gapbs."

    cmd = [bin_path]
    for key, value in kwargs.items():
        if key in ['-h', '-f', '-s', '-g', '-u', '-k', '-m', '-a', '-n', '-r', '-v', '-z', '>']:
            cmd.append(key)
            cmd.append(str(value)) if value else None
        else:
            myLogger.error(f"Unknown option: {key}, try -h for help.")
            sys.exit(1) 

    cmd = " ".join(cmd)
    myLogger.info(f"Running: {cmd}")
    os.chdir(current_file_dir)
    Popen(cmd, shell=True).wait()
    os.chdir(cwd)
