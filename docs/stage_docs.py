#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess


def run(cmd):
    print(cmd)
    subprocess.run([cmd], shell=True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('docs_repo_path', type=str)
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    args = parse_arguments()
    docs_repo_path = os.path.abspath(args.docs_repo_path)

    if not os.path.exists(docs_repo_path):
        run(f"git clone https://github.com/nussl/docs {docs_repo_path}")

    excluded_files = ['.git', '.gitignore', '.nojekyll']
    exclude_args = [f'--exclude {x} ' for x in excluded_files]
    exclude_args = ''.join(exclude_args)
    run(f"rsync -v -r --delete {exclude_args} _build/html/ {docs_repo_path}/")
    run(f"cd {docs_repo_path} && git add . && git commit -am 'updating docs'")
    print(
        f"Inspect what was just committed in {docs_repo_path} \n"
        f"Then run: \n"
        f"\t cd {docs_repo_path} \n"
        f"\t git push origin master"
    )
