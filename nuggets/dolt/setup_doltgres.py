import shutil
import subprocess
import requests


def is_installed():
    return shutil.which("doltgres") is not None

def install():
    if is_installed():
        print("doltgres already installed")
    else:
        r = requests.get('https://github.com/dolthub/doltgresql/releases/latest/download/install.sh')
        r.raise_for_status()
        subprocess.run(["sudo", "bash", "-c", r.text], check=True)


if __name__ == "__main__":
    install()
