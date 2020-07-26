#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import glob
import argparse


def convert_execute_and_sync_script(script_path):
    def run(cmd):
        subprocess.run([cmd], shell=True)

    notebook_path = script_path.replace('py', 'ipynb')
    run(f"jupytext --update --to notebook {script_path}")
    run(f"jupyter nbconvert "
        f"--ExecutePreprocessor.allow_errors=True "
        f"--ExecutePreprocessor.timeout=1000 "
        f"--ExecutePreprocessor.iopub_timeout=5 "
        f"--to notebook --execute --inplace {notebook_path}")
    run(f"jupyter nbconvert {notebook_path}")
    run(f"jupytext --sync {script_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('script_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.script_path == 'all':
        example_scripts = glob.glob("examples/**/*.py", recursive=True)
        for _example_script in example_scripts:
            convert_execute_and_sync_script(_example_script)

        tutorial_scripts = glob.glob("tutorials/**/*.py", recursive=True)
        for _tutorial_script in tutorial_scripts:
            convert_execute_and_sync_script(_tutorial_script)
    else:
        convert_execute_and_sync_script(args.script_path)
