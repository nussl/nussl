#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import glob
import os

def convert_execute_and_sync_script(script_path):
    def run(cmd):
        subprocess.run([cmd], shell=True)
    
    notebook_path = script_path.replace('py', 'ipynb')
    run(f"jupytext --update --to notebook {script_path}")
    run(f"jupyter nbconvert --ExecutePreprocessor.allow_errors=True "
        f"--ExecutePreprocessor.timeout=-1 "
        f"--to notebook --execute --inplace {notebook_path}")
    run(f"jupyter nbconvert {notebook_path}")
    run(f"jupytext --sync {script_path}")

if __name__ == "__main__":
    example_scripts = glob.glob("examples/**/*.py", recursive=True)
    for _example_script in example_scripts:
        convert_execute_and_sync_script(_example_script)

    tutorial_scripts = glob.glob("tutorials/**/*.py", recursive=True)
    for _tutorial_script in tutorial_scripts:
        convert_execute_and_sync_script(_tutorial_script)
