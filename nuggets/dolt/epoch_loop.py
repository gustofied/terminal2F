import setup_doltgres
import subprocess
import time
import json
import psycopg
import concurrent.futures
# rm -rf data .doltcfg && uv run epoch_loop.py

TIMING_FILE = "epoch_timing.json"

NUM_EXAMPLES = 10
ROLLOUTS_PER_EXAMPLE = 4
ROWS_PER_WORKER = 10
NUM_STEPS = 100
TOTAL = NUM_EXAMPLES * ROLLOUTS_PER_EXAMPLE


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


def rollout_worker(args):
    step, ex, r, user, pw, branch = args
    c = psycopg.connect(
        f"host=127.0.0.1 user={user} password={pw} dbname=getting_started/{branch}"
    )
    c.autocommit = True
    for j in range(ROWS_PER_WORKER):
        pid = step * 1000000 + ex * 10000 + r * 100 + j + 1
        c.execute(
            f"INSERT INTO persons VALUES ({pid}, 'S{step}E{ex}', 'R{r}_{j}', 'Addr', 'City')"
        )
    c.execute(
        f"SELECT dolt_commit('-Am', 'step{step} e{ex}_r{r}: {ROWS_PER_WORKER} rows')"
    )
    count = c.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    c.close()
    return count


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

    # table + base seed
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
    base_seed = conn.execute("SELECT commit_hash FROM dolt_log LIMIT 1").fetchone()[0]
    conn.close()

    # create 10 example seeds
    seed_hashes = []
    conn = fresh_conn()
    for ex in range(NUM_EXAMPLES):
        conn.execute(f"SELECT dolt_reset('--hard', '{base_seed}')")
        conn.execute(f"INSERT INTO persons VALUES ({ex * 10000}, 'Seed{ex}', 'Example{ex}', 'SeedAddr', 'SeedCity')")
        conn.execute(f"SELECT dolt_commit('-Am', 'seed for example {ex}')")
        h = conn.execute("SELECT commit_hash FROM dolt_log LIMIT 1").fetchone()[0]
        seed_hashes.append(h)
    conn.execute(f"SELECT dolt_reset('--hard', '{base_seed}')")
    conn.close()

    # create 40 users (once, they persist)
    print(f"creating {TOTAL} users...")
    t0 = time.time()
    for ex in range(NUM_EXAMPLES):
        for r in range(ROLLOUTS_PER_EXAMPLE):
            user = f"agent_e{ex}_r{r}"
            pw = f"{user}pass"
            if not user_exists(user, pw):
                create_user(user, pw)
    t_users = time.time() - t0
    print(f"  users: {t_users:.2f}s")

    # epoch loop
    print(f"\n{'step':>5} | {'create':>7} | {'work':>7} | {'delete':>7} | {'total':>7} | rows/worker")
    print("-" * 65)

    import httpx

    def get_memory_mb():
        try:
            r = httpx.get("http://localhost:11228/metrics", timeout=2)
            for line in r.text.splitlines():
                if line.startswith("go_memstats_alloc_bytes "):
                    return float(line.split()[-1]) / 1024 / 1024
        except Exception:
            pass
        return None

    def get_disk_mb():
        import os
        total = 0
        for dirpath, _, filenames in os.walk("data"):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total / 1024 / 1024

    step_times = []
    timing_data = []
    for step in range(NUM_STEPS):
        # --- create branches ---
        t_create = time.time()
        conn = fresh_conn()
        for ex in range(NUM_EXAMPLES):
            for r in range(ROLLOUTS_PER_EXAMPLE):
                branch = f"e{ex}_r{r}"
                try:
                    conn.execute(f"SELECT dolt_branch('-D', '{branch}')")
                except Exception:
                    pass
                conn.execute(f"SELECT dolt_branch('{branch}', '{seed_hashes[ex]}')")
        conn.close()
        t_create = time.time() - t_create

        # --- parallel work ---
        tasks = []
        for ex in range(NUM_EXAMPLES):
            for r in range(ROLLOUTS_PER_EXAMPLE):
                user = f"agent_e{ex}_r{r}"
                pw = f"{user}pass"
                branch = f"e{ex}_r{r}"
                tasks.append((step, ex, r, user, pw, branch))

        t_work = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=TOTAL) as pool:
            results = list(pool.map(rollout_worker, tasks))
        t_work = time.time() - t_work

        # --- delete branches ---
        t_delete = time.time()
        conn = fresh_conn()
        for ex in range(NUM_EXAMPLES):
            for r in range(ROLLOUTS_PER_EXAMPLE):
                conn.execute(f"SELECT dolt_branch('-D', 'e{ex}_r{r}')")
        conn.close()
        t_delete = time.time() - t_delete

        t_total = t_create + t_work + t_delete
        step_times.append((t_create, t_work, t_delete, t_total))
        mem = get_memory_mb()
        disk = get_disk_mb()

        # verify row counts (seed row + worker rows = 11)
        expected = ROWS_PER_WORKER + 1
        ok = all(c == expected for c in results)
        status = f"{results[0]}" if ok else "MISMATCH"

        print(f"{step:5d} | {t_create:6.2f}s | {t_work:6.2f}s | {t_delete:6.2f}s | {t_total:6.2f}s | {status}")

        # write timing data for live plotting
        timing_data.append({
            "step": step, "create": t_create, "work": t_work,
            "delete": t_delete, "total": t_total,
            "mem_mb": mem, "disk_mb": disk,
        })
        with open(TIMING_FILE, "w") as f:
            json.dump(timing_data, f)

    # summary
    print(f"\n{'='*65}")
    creates = [t[0] for t in step_times]
    works = [t[1] for t in step_times]
    deletes = [t[2] for t in step_times]
    totals = [t[3] for t in step_times]

    def stats(label, vals):
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        first5 = sum(vals[:5]) / min(5, len(vals))
        last5 = sum(vals[-5:]) / min(5, len(vals))
        print(f"  {label:>8}: avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s  first5={first5:.3f}s  last5={last5:.3f}s")

    print(f"  {NUM_STEPS} steps × {TOTAL} rollouts = {NUM_STEPS * TOTAL} branch create/delete cycles")
    print(f"  {NUM_STEPS * TOTAL * ROWS_PER_WORKER} total rows inserted + {NUM_STEPS * TOTAL} commits")
    stats("create", creates)
    stats("work", works)
    stats("delete", deletes)
    stats("total", totals)

    drift = (sum(totals[-5:]) / 5) / (sum(totals[:5]) / 5)
    print(f"\n  drift (last5 / first5): {drift:.2f}x")
    if drift < 1.5:
        print("  verdict: no significant degradation")
    else:
        print(f"  verdict: {drift:.1f}x slowdown over {NUM_STEPS} steps — branch churn may be accumulating")

    # --- POST-RUN: memory and disk investigation ---
    print(f"\n{'='*65}")
    print("  post-run investigation")
    mem_before_gc = get_memory_mb()
    disk_before_gc = get_disk_mb()
    print(f"  before GC:  mem={mem_before_gc:.0f} MB  disk={disk_before_gc:.1f} MB")

    # cooldown — let Go runtime settle
    print("  waiting 10s for cooldown...")
    time.sleep(10)
    mem_after_cool = get_memory_mb()
    print(f"  after cooldown: mem={mem_after_cool:.0f} MB")

    # run dolt_gc
    print("  running dolt_gc()...")
    conn = fresh_conn()
    t_gc = time.time()
    conn.execute("SELECT dolt_gc()")
    t_gc = time.time() - t_gc
    conn.close()
    print(f"  dolt_gc() took {t_gc:.2f}s")

    mem_after_gc = get_memory_mb()
    disk_after_gc = get_disk_mb()
    print(f"  after GC:   mem={mem_after_gc:.0f} MB  disk={disk_after_gc:.1f} MB")
    if disk_before_gc > 0:
        print(f"  disk reclaimed: {disk_before_gc - disk_after_gc:.1f} MB ({(1 - disk_after_gc/disk_before_gc)*100:.0f}%)")

    # append GC data to timing file
    timing_data.append({
        "step": NUM_STEPS, "create": 0, "work": 0, "delete": 0, "total": 0,
        "mem_mb": mem_after_gc, "disk_mb": disk_after_gc,
        "note": "post-GC",
    })
    with open(TIMING_FILE, "w") as f:
        json.dump(timing_data, f)

    proc.terminate()


if __name__ == "__main__":
    main()
