#!/usr/bin/env python3
"""Process popcorn-cli output and return a result summary.

Usage: echo "$RESULT" | python3 tools/_process_result.py <mode> <version> <session_file> <best_file> <tried_file> <kernel_dir> <session_id> [<snapshot_file>]

Prints a one-line [SUBMIT RESULT] message to stdout.
Updates state files as needed.
"""

import sys, re, math, json, datetime, os, subprocess

def geomean_from_text(text, ranked_only=False):
    if ranked_only:
        for label in ('Ranked Benchmark', 'ranked benchmark', 'Ranked benchmark'):
            idx = text.find(label)
            if idx != -1:
                text = text[idx:]
                break
        else:
            return None

    entries = re.findall(r'⏱\s+([0-9]+(?:\.[0-9]+)?)\s*±[^µm\n]*?(µs|ms)', text)
    if not entries:
        entries = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*±[^µm\n]*?(µs|ms)', text)
    if not entries:
        return None

    times_us = []
    for val, unit in entries:
        t = float(val)
        if unit == 'ms':
            t *= 1000.0
        times_us.append(t)
    return math.exp(sum(math.log(t) for t in times_us) / len(times_us))

def log_event(path, event):
    with open(path, 'a') as f:
        f.write(json.dumps(event) + '\n')

def ts():
    return datetime.datetime.now(datetime.UTC).isoformat() + 'Z'

def main():
    mode = sys.argv[1]
    version = int(sys.argv[2])
    session_file = sys.argv[3]
    best_file = sys.argv[4]
    tried_file = sys.argv[5]
    kernel_dir = sys.argv[6]
    session_id = int(sys.argv[7])
    snapshot_file = sys.argv[8] if len(sys.argv) > 8 else None

    result = sys.stdin.read()

    if mode == 'test':
        test_result = 'pass' if re.search(r'pass', result, re.IGNORECASE) else 'fail'
        log_event(session_file, {
            'action': 'submit', 'mode': 'test', 'v': version,
            'result': test_result, 'ts': ts()
        })
        print(f'[SUBMIT RESULT] v{version} test {test_result.upper()}')

    elif mode == 'benchmark':
        score = geomean_from_text(result)
        if score:
            print(f'[SUBMIT RESULT] v{version} benchmark geomean={score:.2f}µs')
        else:
            print(f'[SUBMIT RESULT] v{version} benchmark completed (could not parse geomean)')

    elif mode == 'leaderboard':
        score = geomean_from_text(result, ranked_only=True)

        if score is None:
            log_event(session_file, {
                'action': 'submit', 'mode': 'leaderboard', 'v': version,
                'score': None, 'best': None, 'kept': False,
                'reason': 'could not parse score', 'ts': ts()
            })
            subprocess.run(['cp', os.path.join(kernel_dir, 'best_submission.py'),
                          os.path.join(kernel_dir, 'submission.py')])
            print(f'[SUBMIT RESULT] v{version} leaderboard REVERT (could not parse score)')
            return

        best = json.load(open(best_file))
        best_score = best['score']
        best_version = best['version']
        improved = score < best_score * 0.999

        # Get latest proposal
        prop = None
        with open(session_file) as f:
            for line in f:
                e = json.loads(line)
                if e.get('action') == 'propose':
                    prop = e
        prop_what = prop['what'] if prop else 'unknown'
        prop_keywords = prop['keywords'] if prop else []

        if improved:
            change = f'{(score / best_score - 1) * 100:.1f}%'
            json.dump({'version': version, 'score': score, 'ts': ts()}, open(best_file, 'w'))
            keep_src = snapshot_file if snapshot_file else os.path.join(kernel_dir, 'submission.py')
            subprocess.run(['cp', keep_src, os.path.join(kernel_dir, 'best_submission.py')])
            log_event(tried_file, {
                'v': version, 'what': prop_what, 'keywords': prop_keywords,
                'score': score, 'kept': True, 'reason': 'improved geomean',
                'session': session_id
            })
            subprocess.run(['git', 'add', '-A'], cwd=kernel_dir)
            subprocess.run(['git', 'commit', '-m', f'v{version}: KEEP — {prop_what} ({change})'],
                         cwd=kernel_dir, capture_output=True)
        else:
            change = f'{(score / best_score - 1) * 100:+.1f}%'
            subprocess.run(['cp', os.path.join(kernel_dir, 'best_submission.py'),
                          os.path.join(kernel_dir, 'submission.py')])
            log_event(tried_file, {
                'v': version, 'what': prop_what, 'keywords': prop_keywords,
                'score': score, 'kept': False, 'reason': f'{change} vs best',
                'session': session_id
            })

        # Log to session file before counting reverts
        log_event(session_file, {
            'action': 'submit', 'mode': 'leaderboard', 'v': version,
            'score': score, 'best': best_score, 'kept': improved, 'ts': ts()
        })

        # Print result message
        if improved:
            print(f'[SUBMIT RESULT] v{version} leaderboard KEEP {score:.2f}µs ({change} vs v{best_version} @ {best_score}µs)')
        else:
            revert_count = 0
            with open(session_file) as f:
                for line in f:
                    e = json.loads(line)
                    if e.get('action') == 'submit' and e.get('mode') == 'leaderboard' and e.get('kept') is False:
                        revert_count += 1
            msg = f'[SUBMIT RESULT] v{version} leaderboard REVERT {score:.2f}µs ({change} vs v{best_version} @ {best_score}µs) [{revert_count}/5 reverts]'
            if revert_count >= 5:
                msg += ' — REVERT LIMIT REACHED. Run: ./tools/close_branch.sh --what-failed "summary" && exit'
            print(msg)

if __name__ == '__main__':
    main()
