import setup_doltgres
import subprocess
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


def fresh_conn(dbname="getting_started"):
    conn = psycopg.connect(f"host=127.0.0.1 user=postgres password=password dbname={dbname}")
    conn.autocommit = True
    return conn


def user_exists(name, pw):
    try:
        c = psycopg.connect(f"host=127.0.0.1 user={name} password={pw} dbname=getting_started")
        c.close()
        return True
    except Exception:
        return False


def create_user(name, pw):
    conn = fresh_conn()
    conn.execute(f"CREATE USER {name} WITH PASSWORD '{pw}'")
    conn.close()
    conn = fresh_conn()
    conn.execute(f"GRANT ALL ON DATABASE getting_started TO {name}")
    conn.close()
    conn = fresh_conn()
    conn.execute(f"GRANT ALL ON SCHEMA public TO {name}")
    conn.close()
    conn = fresh_conn()
    conn.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {name}")
    conn.close()


def main():
    setup_doltgres.install()

    subprocess.run(["pkill", "-x", "doltgres"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    proc = subprocess.Popen(["doltgres"])
    time.sleep(2)

    # database
    try:
        fresh_conn()
    except Exception:
        conn = fresh_conn("postgres")
        conn.execute("CREATE DATABASE IF NOT EXISTS getting_started")
        conn.close()

    # table
    conn = fresh_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS persons (
        PersonID int PRIMARY KEY,
        LastName varchar(255) NOT NULL,
        FirstName varchar(255),
        Address varchar(255),
        City varchar(255))""")
    try:
        conn.execute("SELECT dolt_commit('-Am', 'create persons table')")
    except psycopg.errors.InternalError_:
        pass
    conn.close()

    # users
    if not user_exists("agent1", "agent1pass"):
        create_user("agent1", "agent1pass")
        print("  created agent1")

    if not user_exists("agent2", "agent2pass"):
        create_user("agent2", "agent2pass")
        print("  created agent2")

    # agent1 writes
    print("\n=== agent1 writes ===")
    a1 = psycopg.connect("host=127.0.0.1 user=agent1 password=agent1pass dbname=getting_started")
    a1.autocommit = True
    a1.execute("INSERT INTO persons VALUES (1, 'Smith', 'Alice', '123 Main St', 'Oslo') ON CONFLICT DO NOTHING")
    try:
        a1.execute("SELECT dolt_commit('-Am', 'agent1: add Alice Smith')")
    except psycopg.errors.InternalError_:
        pass
    print("  done")
    a1.close()

    # agent2 writes
    print("\n=== agent2 writes ===")
    a2 = psycopg.connect("host=127.0.0.1 user=agent2 password=agent2pass dbname=getting_started")
    a2.autocommit = True
    a2.execute("INSERT INTO persons VALUES (2, 'Jones', 'Bob', '456 Oak Ave', 'Bergen') ON CONFLICT DO NOTHING")
    try:
        a2.execute("SELECT dolt_commit('-Am', 'agent2: add Bob Jones')")
    except psycopg.errors.InternalError_:
        pass
    print("  done")
    a2.close()

    # attribution
    conn = fresh_conn()
    print("\n=== persons ===")
    print_table(conn.execute("SELECT * FROM persons"))

    print("\n=== dolt_log (attribution) ===")
    print_table(conn.execute("SELECT commit_hash, committer, message FROM dolt_log"))

    conn.close()
    proc.terminate()


if __name__ == "__main__":
    main()
