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

    # save seed commit hash — this is the fixed starting point for resets
    seed_hash = conn.execute("SELECT commit_hash FROM dolt_log LIMIT 1").fetchone()[0]
    print(f"  seed: {seed_hash[:8]}")
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

    # --- MERGE ---
    # merge dev1 into main as postgres (the orchestrator)
    print("\n=== merge dev1 into main ===")
    conn = fresh_conn()
    conn.execute("SELECT dolt_merge('dev1')")
    try:
        conn.execute("SELECT dolt_commit('-Am', 'merge dev1 into main')")
    except psycopg.errors.InternalError_:
        pass
    print("  persons on main after merge:")
    print_table(conn.execute("SELECT * FROM persons"))
    print("  dolt_log:")
    print_table(conn.execute("SELECT commit_hash, committer, message FROM dolt_log"))
    conn.close()

    # --- RESET FROM MAIN (current promoted state) ---
    print("\n=== reset dev1 from main (has Alice after merge) ===")
    conn = fresh_conn()
    conn.execute("SELECT dolt_branch('-D', 'dev1')")
    conn.execute("SELECT dolt_branch('dev1', 'main')")
    conn.close()

    a1 = psycopg.connect("host=127.0.0.1 user=agent1 password=agent1pass dbname=getting_started/dev1")
    a1.autocommit = True
    print("  persons on dev1 (from main):")
    print_table(a1.execute("SELECT * FROM persons"))
    a1.close()

    # --- RESET FROM SEED (original empty table) ---
    print(f"\n=== reset dev1 from seed ({seed_hash[:8]}) ===")
    conn = fresh_conn()
    conn.execute("SELECT dolt_branch('-D', 'dev1')")
    conn.execute(f"SELECT dolt_branch('dev1', '{seed_hash}')")
    conn.close()

    a1 = psycopg.connect("host=127.0.0.1 user=agent1 password=agent1pass dbname=getting_started/dev1")
    a1.autocommit = True
    print("  persons on dev1 (from seed — should be empty):")
    print_table(a1.execute("SELECT * FROM persons"))

    # agent does fresh work from seed
    a1.execute("INSERT INTO persons VALUES (3, 'Park', 'Charlie', '789 Pine Rd', 'Trondheim')")
    a1.execute("SELECT dolt_commit('-Am', 'agent1: add Charlie Park')")
    print("  after new work:")
    print_table(a1.execute("SELECT * FROM persons"))
    print("  dev1 log:")
    print_table(a1.execute("SELECT commit_hash, committer, message FROM dolt_log"))
    a1.close()

    # main unaffected
    print("\n=== main (should still only have Alice) ===")
    conn = fresh_conn()
    print_table(conn.execute("SELECT * FROM persons"))
    conn.close()

    # --- 40-ROLLOUT CONCURRENCY TEST ---
    # 10 examples × 4 rollouts = 40 parallel workers
    # each example has its own seed state, 4 rollouts fork from that seed
    import concurrent.futures

    NUM_EXAMPLES = 10
    ROLLOUTS_PER_EXAMPLE = 4
    ROWS_PER_WORKER = 10
    TOTAL = NUM_EXAMPLES * ROLLOUTS_PER_EXAMPLE

    print(f"\n=== 40-rollout test ({NUM_EXAMPLES} examples × {ROLLOUTS_PER_EXAMPLE} rollouts, {ROWS_PER_WORKER} rows each) ===")

    # create 10 different seed states on main
    # each seed has a different pre-loaded row to distinguish them
    seed_hashes = []
    conn = fresh_conn()
    for ex in range(NUM_EXAMPLES):
        # reset main to original seed (empty table)
        conn.execute(f"SELECT dolt_reset('--hard', '{seed_hash}')")
        # insert a unique seed row for this example
        conn.execute(f"INSERT INTO persons VALUES ({ex * 10000}, 'Seed{ex}', 'Example{ex}', 'SeedAddr', 'SeedCity')")
        conn.execute(f"SELECT dolt_commit('-Am', 'seed for example {ex}')")
        h = conn.execute("SELECT commit_hash FROM dolt_log LIMIT 1").fetchone()[0]
        seed_hashes.append(h)
        print(f"  seed[{ex}]: {h[:8]}")
    # restore main to original seed
    conn.execute(f"SELECT dolt_reset('--hard', '{seed_hash}')")
    conn.close()

    # create 40 users and 40 branches
    print("  creating users and branches...")
    conn = fresh_conn()
    tasks = []
    for ex in range(NUM_EXAMPLES):
        for r in range(ROLLOUTS_PER_EXAMPLE):
            user = f"agent_e{ex}_r{r}"
            pw = f"{user}pass"
            branch = f"e{ex}_r{r}"

            if not user_exists(user, pw):
                create_user(user, pw)

            try:
                conn.execute(f"SELECT dolt_branch('-D', '{branch}')")
            except Exception:
                pass
            conn.execute(f"SELECT dolt_branch('{branch}', '{seed_hashes[ex]}')")

            tasks.append((ex, r, user, pw, branch))
    conn.close()
    print(f"  {len(tasks)} workers ready")

    # worker function
    def rollout_worker(args):
        ex, r, user, pw, branch = args
        c = psycopg.connect(
            f"host=127.0.0.1 user={user} password={pw} dbname=getting_started/{branch}"
        )
        c.autocommit = True
        for j in range(ROWS_PER_WORKER):
            pid = ex * 10000 + r * 100 + j + 1  # +1 to avoid collision with seed row
            c.execute(
                f"INSERT INTO persons VALUES ({pid}, 'E{ex}', 'R{r}_{j}', 'Addr{j}', 'City{j}')"
            )
        c.execute(
            f"SELECT dolt_commit('-Am', 'e{ex}_r{r}: {ROWS_PER_WORKER} rows')"
        )
        count = c.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        log = c.execute("SELECT committer, message FROM dolt_log LIMIT 1").fetchone()
        c.close()
        return ex, r, branch, count, log

    # run all 40 in parallel
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=TOTAL) as pool:
        results = list(pool.map(rollout_worker, tasks))
    elapsed = time.time() - t0

    # summary
    print(f"\n=== results ({elapsed:.2f}s) ===")
    for ex, r, branch, count, log in sorted(results):
        print(f"  {branch}: {count} rows | {log[0]} | {log[1]}")

    # --- VERIFICATION ---
    print("\n=== verification ===")
    errors = []

    # 1) each rollout should have exactly ROWS_PER_WORKER + 1 (seed row + worker rows)
    expected_count = ROWS_PER_WORKER + 1
    for ex, r, branch, count, log in results:
        if count != expected_count:
            errors.append(f"  FAIL {branch}: expected {expected_count} rows, got {count}")

    # 2) all 4 rollouts from same example should share the same seed row
    for ex in range(NUM_EXAMPLES):
        seed_pid = ex * 10000
        for r in range(ROLLOUTS_PER_EXAMPLE):
            branch = f"e{ex}_r{r}"
            c = fresh_conn(f"getting_started/{branch}")
            row = c.execute(f"SELECT * FROM persons WHERE PersonID = {seed_pid}").fetchone()
            if row is None:
                errors.append(f"  FAIL {branch}: missing seed row {seed_pid}")
            elif row[1] != f"Seed{ex}":
                errors.append(f"  FAIL {branch}: wrong seed row, got {row[1]}")
            c.close()

    # 3) rollouts from different examples should NOT have each other's seed rows
    for ex in range(NUM_EXAMPLES):
        branch = f"e{ex}_r0"  # check first rollout of each example
        c = fresh_conn(f"getting_started/{branch}")
        for other_ex in range(NUM_EXAMPLES):
            if other_ex == ex:
                continue
            other_pid = other_ex * 10000
            row = c.execute(f"SELECT * FROM persons WHERE PersonID = {other_pid}").fetchone()
            if row is not None:
                errors.append(f"  FAIL {branch}: has seed row from example {other_ex}")
        c.close()

    # 4) attribution — each rollout committed by its own user
    for ex, r, branch, count, log in results:
        expected_user = f"agent_e{ex}_r{r}"
        if log[0] != expected_user:
            errors.append(f"  FAIL {branch}: committed by {log[0]}, expected {expected_user}")

    # 5) main should be back at seed (empty table, since we reset main after creating seeds)
    conn = fresh_conn()
    main_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    conn.close()
    if main_count != 0:
        errors.append(f"  FAIL main: expected 0 rows (reset to seed), got {main_count}")

    if errors:
        print("  ERRORS:")
        for e in errors:
            print(e)
    else:
        print(f"  ALL PASSED: {TOTAL} rollouts, {TOTAL * ROWS_PER_WORKER} new rows, {NUM_EXAMPLES} seeds verified")
        print(f"  isolation: OK | attribution: OK | main untouched: OK")

    proc.terminate()


if __name__ == "__main__":
    main()
