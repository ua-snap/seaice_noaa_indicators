"""Render the jupyter notebooks used for project updates"""

import argparse, subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a .ipynb to html using nbconvert (for project updates)"
    )
    parser.add_argument(
        "-f",
        "--file-path",
        action="store",
        dest="fp",
        type=str,
        help=("Path to file to be converted"),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        action="store",
        dest="out_dir",
        type=str,
        help=("Directory to save converted nb"),
    )
    parser.add_argument(
        "-of",
        "--output-format",
        action="store",
        dest="out_format",
        type=str,
        default="html",
        help=("Output format of notebook"),
    )

    # unpack args
    args = parser.parse_args()
    fp = args.fp
    out_dir = args.out_dir
    out_format = args.out_format
    
    command = [
        "jupyter",
        "nbconvert",
        fp,
        "--output-dir",
        out_dir,
        "--to",
        out_format,
        "--no-prompt",
        "--no-input",
        "--execute",
    ]
    
    if out_format == "html":
        command.append("--template")
        command.append("classic")
    
    subprocess.Popen(command)
