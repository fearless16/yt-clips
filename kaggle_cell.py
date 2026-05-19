def run(cmd, check=False, timeout=60, capture=True):
    import subprocess
    try:
        r = subprocess.run(cmd, shell=True, capture_output=capture, text=True, timeout=timeout)
        if check and r.returncode != 0:
            print(f"  \u274c FAILED ({r.returncode}): {cmd[:80]}")
            if r.stderr: print(f"     {r.stderr.strip()[-200:]}")
            return None
        return r
    except subprocess.TimeoutExpired:
        print(f"  \u23f0 TIMEOUT (>={timeout}s): {cmd[:80]}")
        return None
    except Exception as e:
        print(f"  \u274c ERROR: {e}")
        return None

def kill_old_processes() -> None:
    run("pkill -f 'python watcher.py' 2>/dev/null || true")
    run("pkill -f 'lt --port' 2>/dev/null || true")
    run("pkill -f 'localtunnel' 2>/dev/null || true")
    run("fuser -k 5000/tcp 2>/dev/null || true")
    run("lsof -ti:5000 | xargs kill -9 2>/dev/null || true")
    time.sleep(2)

def install_lt():
    """Install localtunnel, try multiple methods."""
    # Check if already available
    r = run("which lt 2>/dev/null || command -v lt 2>/dev/null")
    if r and r.stdout.strip():
        print(f'  \u2705 localtunnel found: {r.stdout.strip()}')
        return True
    
    print('  Installing localtunnel...')
    run("npm install -g localtunnel 2>&1", timeout=60)
    run("npm install -g localtunnel@2.0.2 2>&1", timeout=60)  # fallback stable version
    
    # Check again
    r = run("which lt 2>/dev/null || command -v lt 2>/dev/null || echo 'NOT_FOUND'")
    if r and r.stdout.strip() and r.stdout.strip() != 'NOT_FOUND':
        print(f'  \u2705 localtunnel installed: {r.stdout.strip()}')
        return True
    
    # Try npx fallback
    r = run("npx --yes localtunnel --version 2>&1")
    if r and r.returncode == 0:
        print('  \u2705 localtunnel available via npx')
        return True
    
    print('  \u274c localtunnel install failed')
    return False

def start_services() -> None:
    import os, time, subprocess
    from pathlib import Path
    
    print('\u2550' * 50)
    print('  \U0001f3e0 yt-clips Worker Setup')
    print('\u2550' * 50)
    
    kill_old_processes()
    
    # Git pull
    print('\n  1/5 Pulling latest code...')
    r = run('git pull origin main 2>&1', timeout=30)
    if r is None:
        print('  \u26a0\ufe0f  Git pull failed')
    elif 'Already up to date' in r.stdout:
        print('  \u2705 Already up to date')
    elif r.returncode == 0:
        print(f'  \u2705 Pulled')

    # Install localtunnel
    print('\n  2/5 Setting up tunnel...')
    lt_ok = install_lt()
    
    # Create folders
    print('\n  3/5 Creating folders...')
    for folder in ["input", "temp", "transcripts", "highlights", "shorts", "logs", "photos"]:
        Path(folder).mkdir(exist_ok=True)
    print('  \u2705 Done')

    # Start watcher
    print('\n  4/5 Starting watcher...')
    watcher = subprocess.Popen(
        ["nohup", "python", "watcher.py"],
        stdout=open("watcher.log","w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(3)
    
    check = run("pgrep -f 'python watcher.py'")
    if check and check.stdout.strip():
        pid = check.stdout.strip().split()[0]
        print(f'  \u2705 Watcher running (PID: {pid})')
    else:
        print('  \u274c Watcher failed! Log:')
        for l in open("watcher.log").read().strip().splitlines()[-5:]:
            print(f'     {l}')
        return

    # Start tunnel
    print('\n  5/5 Starting tunnel...')
    
    if not lt_ok:
        # Try npx fallback directly
        tunnel = subprocess.Popen(
            ["nohup", "npx", "--yes", "localtunnel", "--port", "5000"],
            stdout=open("tunnel.log","w"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    else:
        tunnel = subprocess.Popen(
            ["nohup", "lt", "--port", "5000"],
            stdout=open("tunnel.log","w"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(5)

    # Check tunnel process
    check_t = run("pgrep -af 'lt --port|localtunnel' 2>/dev/null")
    if check_t and check_t.stdout.strip():
        print(f'  \u2705 Tunnel process running')
    else:
        print('  \u274c Tunnel process died!')
        # Show what's in log
        content = open("tunnel.log").read().strip()
        if content:
            print(f'     Log: {content[-500:]}')
        else:
            print('     Log is EMPTY')
            # Try running in foreground to see error
            print('  Diagnosing...')
            r = run("lt --port 5000 2>&1", timeout=10, capture=True)
            if r:
                print(f'     stdout: {r.stdout.strip()[-200:]}')
                print(f'     stderr: {r.stderr.strip()[-200:]}')
                print(f'     exit: {r.returncode}')
        return

    # Wait for URL with better parsing
    print('  Waiting for tunnel URL...')
    url = None
    for i in range(30):
        time.sleep(2)
        content = open("tunnel.log").read()
        # Try multiple patterns
        for line in content.splitlines():
            line = line.strip()
            if "://" in line:
                # Extract URL (sometimes it has extra text)
                for word in line.split():
                    if "://" in word and "loca.lt" in word:
                        url = word.strip().rstrip(',.')
                        break
            if url:
                break
        if url:
            break
        if i % 5 == 0:
            print(f'     ...waiting ({i*2+2}s)')
    
    if url:
        Path("kaggle_url.txt").write_text(url)
        print(f'  \u2705 Tunnel URL: {url}')
        print('\n' + '\u2550' * 50)
        print('  \u2705 WORKER IS ONLINE!')
        print('\u2550' * 50)
        print('\n  Mac pe chalao:')
        print(f'  python bridge.py "https://youtu.be/VIDEO_ID"')
    else:
        print('  \u26a0\ufe0f  Tunnel URL not found. tunnel.log:')
        for l in open("tunnel.log").read().strip().splitlines()[-8:]:
            print(f'     {l}')
        print('\n  \U0001f447 Khud check karo:')
        print('  !cat tunnel.log')
        print('  !lt --port 5000 --print-requests')

start_services()
