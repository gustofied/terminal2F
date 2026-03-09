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

    conn.execute("""CREATE TABLE IF NOT EXISTS employees (
        id int8, last_name text, first_name text, primary key(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS teams (
        id int8, team_name text, primary key(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS employees_teams (
        team_id int8, employee_id int8, primary key(team_id, employee_id),
        foreign key (team_id) references teams(id),
        foreign key (employee_id) references employees(id))""")

    conn.execute("SELECT dolt_add('teams', 'employees', 'employees_teams')")
    try:
        conn.execute("SELECT dolt_commit('-m', 'Created initial schema')")
    except Exception as e:
        if "nothing to commit" in str(e):
            print("Nothing to commit, schema already exists")
        else:
            raise

    # Insert an employee and commit
    conn.execute("INSERT INTO employees VALUES (1, 'Adams', 'John') ON CONFLICT DO NOTHING")
    conn.execute("SELECT dolt_commit('-am', 'Added John Adams')")

    # Update the employee and commit
    conn.execute("UPDATE employees SET first_name = 'Jonathan' WHERE id = 1")
    conn.execute("SELECT dolt_commit('-am', 'Changed John to Jonathan')")

    # Update again and commit
    conn.execute("UPDATE employees SET last_name = 'Adamson' WHERE id = 1")
    conn.execute("SELECT dolt_commit('-am', 'Changed Adams to Adamson')")

    # Now check the full history of this specific row
    print("\n=== history of employee id=1 ===")
    print_table(conn.execute("""
        SELECT id, first_name, last_name, commit_hash, commit_date
        FROM dolt_history_employees
        WHERE id = 1
        ORDER BY commit_date
    """))

    print("\n=== dolt_log ===")
    print_table(conn.execute("SELECT * FROM dolt_log"))

    conn.close()
    proc.terminate()


if __name__ == "__main__":
    main()
