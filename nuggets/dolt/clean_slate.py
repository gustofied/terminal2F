import setup_doltgres
import subprocess
import signal
import time
import psycopg


def print_table(cursor):
    rows = cursor.fetchall()
    if not rows:
        print("(empty)")
        return
    cols = [desc[0] for desc in cursor.description]
    widths = [len(c) for c in cols]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in widths)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)))

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

    conn = psycopg.connect("host=127.0.0.1 user=postgres password=password dbname=getting_started")
    conn.autocommit = True

    print("\n=== dolt_branches ===")
    print_table(conn.execute("SELECT * FROM dolt_branches"))

    print("\n=== dolt_branch_control ===")
    print_table(conn.execute("SELECT * FROM dolt_branch_control"))

    print("\n=== dolt_log ===")
    print_table(conn.execute("SELECT * FROM dolt_log"))

    conn.close()
    proc.terminate()


if __name__ == "__main__":
    main()
