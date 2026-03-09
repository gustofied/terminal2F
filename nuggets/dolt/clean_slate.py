import setup_doltgres
import subprocess
import time
import shutil
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


def fresh_conn(dbname="getting_started"):
    conn = psycopg.connect(f"host=127.0.0.1 user=postgres password=password dbname={dbname}")
    conn.autocommit = True
    return conn


def main():
    setup_doltgres.install()

    # kill any running doltgres and wipe data for a true clean slate
    subprocess.run(["pkill", "-x", "doltgres"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    for d in ["data", ".doltcfg"]:
        shutil.rmtree(d, ignore_errors=True)

    proc = subprocess.Popen(["doltgres"])
    time.sleep(2)

    # create database
    conn = fresh_conn("postgres")
    conn.execute("CREATE DATABASE IF NOT EXISTS getting_started")
    conn.close()

    # create table
    conn = fresh_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS persons (
        PersonID int PRIMARY KEY,
        LastName varchar(255) NOT NULL,
        FirstName varchar(255),
        Address varchar(255),
        City varchar(255))""")
    conn.execute("SELECT dolt_commit('-Am', 'create persons table')")
    conn.close()

    # create agent1
    conn = fresh_conn()
    conn.execute("CREATE USER agent1 WITH PASSWORD 'agent1pass'")
    conn.close()
    conn = fresh_conn()
    conn.execute("GRANT ALL ON DATABASE getting_started TO agent1")
    conn.close()
    conn = fresh_conn()
    conn.execute("GRANT ALL ON SCHEMA public TO agent1")
    conn.close()
    conn = fresh_conn()
    conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO agent1")
    conn.close()

    # create agent2
    conn = fresh_conn()
    conn.execute("CREATE USER agent2 WITH PASSWORD 'agent2pass'")
    conn.close()
    conn = fresh_conn()
    conn.execute("GRANT ALL ON DATABASE getting_started TO agent2")
    conn.close()
    conn = fresh_conn()
    conn.execute("GRANT ALL ON SCHEMA public TO agent2")
    conn.close()
    conn = fresh_conn()
    conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO agent2")
    conn.close()

    # verify: list users and log
    conn = fresh_conn()
    print("\n=== verify users ===")
    for name, pw in [("agent1", "agent1pass"), ("agent2", "agent2pass")]:
        try:
            c = psycopg.connect(f"host=127.0.0.1 user={name} password={pw} dbname=getting_started")
            c.autocommit = True
            c.execute("SELECT 1")
            print(f"  {name}: login OK")
            c.close()
        except Exception as e:
            print(f"  {name}: {e}")

    # agent1 inserts a row and commits
    print("\n=== agent1 writes ===")
    a1 = psycopg.connect("host=127.0.0.1 user=agent1 password=agent1pass dbname=getting_started")
    a1.autocommit = True
    a1.execute("INSERT INTO persons VALUES (1, 'Smith', 'Alice', '123 Main St', 'Oslo')")
    a1.execute("SELECT dolt_commit('-Am', 'agent1: add Alice Smith')")
    print("  agent1 committed")
    a1.close()

    # agent2 inserts a row and commits
    print("\n=== agent2 writes ===")
    a2 = psycopg.connect("host=127.0.0.1 user=agent2 password=agent2pass dbname=getting_started")
    a2.autocommit = True
    a2.execute("INSERT INTO persons VALUES (2, 'Jones', 'Bob', '456 Oak Ave', 'Bergen')")
    a2.execute("SELECT dolt_commit('-Am', 'agent2: add Bob Jones')")
    print("  agent2 committed")
    a2.close()

    # see who did what
    print("\n=== persons ===")
    conn = fresh_conn()
    print_table(conn.execute("SELECT * FROM persons"))

    print("\n=== dolt_log (attribution) ===")
    print_table(conn.execute("SELECT commit_hash, committer, message FROM dolt_log"))

    conn.close()
    proc.terminate()


if __name__ == "__main__":
    main()
