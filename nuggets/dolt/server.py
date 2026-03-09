import setup_doltgres
import subprocess
import signal
import time
import psycopg

def kill_existing():
    subprocess.run(["pkill", "-x", "doltgres"], stderr=subprocess.DEVNULL)
    time.sleep(1)

def main():
    setup_doltgres.install()
    kill_existing()
    proc = subprocess.Popen(["doltgres"])
    time.sleep(2)

    conn = psycopg.connect("host=127.0.0.1 user=postgres password=password dbname=postgres")
    conn.autocommit = True
    conn.execute("CREATE DATABASE IF NOT EXISTS getting_started")
    conn.close()

    print("server running on localhost:5432")
    print("databases: postgres, getting_started")
    print("ctrl+c to stop\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nserver stopped")

if __name__ == "__main__":
    main()
