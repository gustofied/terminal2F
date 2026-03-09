import setup_doltgres
import subprocess
import time
import psycopg
# rm -rf data .doltcfg && uv run clean_slate.py

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
    conn.execute(f"GRANT ALL ON DATABASE getting_started TO {name}")
    conn.execute(f"GRANT ALL ON SCHEMA public TO {name}")
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

    # branches
    conn = fresh_conn()
    try:
        conn.execute("SELECT dolt_branch('dev1', 'main')")
    except Exception:
        pass
    try:
        conn.execute("SELECT dolt_branch('dev2', 'main')")
    except Exception:
        pass
    conn.close()

    # agent1 writes on dev1
    print("\n=== agent1 on dev1 ===")
    a1 = psycopg.connect("host=127.0.0.1 user=agent1 password=agent1pass dbname=getting_started/dev1")
    a1.autocommit = True
    a1.execute("INSERT INTO persons VALUES (1, 'Smith', 'Alice', '123 Main St', 'Oslo') ON CONFLICT DO NOTHING")
    try:
        a1.execute("SELECT dolt_commit('-Am', 'agent1: add Alice Smith')")
    except psycopg.errors.InternalError_:
        pass
    print("  persons on dev1:")
    print_table(a1.execute("SELECT * FROM persons"))
    a1.close()

    # agent2 writes on dev2
    print("\n=== agent2 on dev2 ===")
    a2 = psycopg.connect("host=127.0.0.1 user=agent2 password=agent2pass dbname=getting_started/dev2")
    a2.autocommit = True
    a2.execute("INSERT INTO persons VALUES (2, 'Jones', 'Bob', '456 Oak Ave', 'Bergen') ON CONFLICT DO NOTHING")
    try:
        a2.execute("SELECT dolt_commit('-Am', 'agent2: add Bob Jones')")
    except psycopg.errors.InternalError_:
        pass
    print("  persons on dev2:")
    print_table(a2.execute("SELECT * FROM persons"))
    a2.close()

    # main should be clean
    print("\n=== main (should have no agent data) ===")
    conn = fresh_conn()
    print("  persons on main:")
    print_table(conn.execute("SELECT * FROM persons"))

    # all branches
    print("\n=== branches ===")
    print_table(conn.execute("SELECT name, hash FROM dolt_branches"))

    # attribution across branches
    print("\n=== dolt_log on main ===")
    print_table(conn.execute("SELECT commit_hash, committer, message FROM dolt_log"))
    conn.close()

    # check dev1 log
    print("\n=== dolt_log on dev1 ===")
    c1 = psycopg.connect("host=127.0.0.1 user=agent1 password=agent1pass dbname=getting_started/dev1")
    c1.autocommit = True
    print_table(c1.execute("SELECT commit_hash, committer, message FROM dolt_log"))
    c1.close()

    # check dev2 log
    print("\n=== dolt_log on dev2 ===")
    c2 = psycopg.connect("host=127.0.0.1 user=agent2 password=agent2pass dbname=getting_started/dev2")
    c2.autocommit = True
    print_table(c2.execute("SELECT commit_hash, committer, message FROM dolt_log"))
    c2.close()

    # test dolt_branch_namespace_control INSERT
    print("\n=== dolt_branch_namespace_control ===")
    conn = fresh_conn()
    try:
        conn.execute("INSERT INTO dolt_branch_namespace_control VALUES ('%', 'dev%', 'agent1', '%')")
        print("  OK")
    except Exception as e:
        print(f"  {e}")
    conn.close()

    proc.terminate()


if __name__ == "__main__":
    main()
