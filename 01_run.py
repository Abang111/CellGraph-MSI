#!/usr/bin/env python
"""
v32_final / 01_run.py
Re-run v17's concept-supervised model, three seeds, on two named GPUs, resumably.

WHAT IS AND IS NOT CHANGED

    The trainer is v32_final/_gen_train_v17.py, produced by v22/02_retrain_v17.py
    from v17/02_train_gnn.py with two additions and nothing else: the best epoch's
    weights are kept, and the checkpoint is written afterwards. Eleven invariants
    covering the training semantics are asserted by that patcher. Every
    hyperparameter here is the v17 default -- 48 windows, 30 epochs, batch 4,
    lr 3e-4, hidden 96, three layers, concept weight 0.3, COAD only -- so nothing
    is passed that could silently differ from the published run.

    The epoch is selected on the fold being scored. That is v17's protocol and the
    protocol the dissertation describes; it is optimistic and the resulting figure
    must not be placed beside an inner-selected number.

WHY EACH SEED GETS ITS OWN OUTPUT DIRECTORY

    The trainer writes its CSVs to {out_dir}/{tag}/, a path with no seed in it,
    while checkpoints go to {out_dir}/{tag}_seed{n}/. The previous run launched
    one process per seed against a shared out_dir, so five runs overwrote one
    another's oof_predictions.csv and only the last seed survived. The
    checkpoints were unaffected, which is why 25 of them exist alongside a single
    seed's predictions. Giving each seed its own out_dir removes the collision
    without touching the trainer.

RESUME IS PER SEED, NOT PER FOLD

    fold_records.csv is written once, after all five folds of a seed finish, so
    its presence with five rows is a reliable completion marker. A seed that died
    part-way leaves checkpoints but no CSV and is restarted from fold zero: the
    trainer loops folds internally and has no --folds argument, so there is
    nowhere to re-enter. About fifty minutes is the most that can be lost.

USAGE
    python v32_final/01_run.py --dry-run
    python v32_final/01_run.py --gpus 5 7
    python v32_final/01_run.py --gpus 5 7          # again, to resume
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TRAINER = Path("v32_final/_gen_train_v17.py")

# pooled out-of-fold AUROC from v17/train_L1_seed*.log. GPU arithmetic is not
# deterministic, so these are targets to land near, not values to match.
RECORDED_POOLED = {0: 0.870, 1: 0.881, 2: 0.891, 3: 0.895, 4: 0.889}
TOLERANCE = 0.02


def seed_dir(out_root, seed):
    return Path(out_root) / f"seed{seed}"


def status(out_root, seed, tag):
    """done / partial / pending, from what is on disk."""
    d = seed_dir(out_root, seed)
    csv = d / tag / "fold_records.csv"
    cks = sorted((d / f"{tag}_seed{seed}").glob("*.pt"))
    if csv.exists():
        try:
            import pandas as pd
            n = len(pd.read_csv(csv))
        except Exception:
            n = 0
        if n >= 5:
            return "done", len(cks), n
    return ("partial" if cks else "pending"), len(cks), 0


def launch(seed, gpu, out_root, tag, trainer, extra):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # the trainer sees one visible device, so it must address it as cuda:0
    cmd = [sys.executable, "-u", str(trainer),
           "--tag", tag, "--seeds", str(seed),
           "--out-dir", str(seed_dir(out_root, seed))] + extra
    log = Path(out_root).parent / f"log_{tag}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p, fh, log


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gpus", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--tag", default="L1_concept")
    ap.add_argument("--out-root", default="v32_final/runs")
    ap.add_argument("--trainer", default=str(TRAINER))
    ap.add_argument("--force", action="store_true",
                    help="rerun seeds already marked done")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="passed through to the trainer; leave empty to keep "
                         "every v17 default")
    args = ap.parse_args()

    trainer = Path(args.trainer)
    if not trainer.exists():
        sys.exit(f"[fatal] {trainer} not found. Generate it first:\n"
                 f"          python v22/02_retrain_v17.py --dry-run\n"
                 f"          cp v22/_gen_train_v17.py {trainer}")

    print(f"trainer  {trainer}")
    print(f"tag      {args.tag}    selection: holdout_best (v17 protocol)")
    print(f"out      {args.out_root}/seed<n>/")
    print(f"gpus     {args.gpus}")
    if args.extra:
        print(f"extra    {args.extra}   <-- not a v17 default; check this is "
              f"intended")
    print()

    todo = []
    for s in args.seeds:
        st, n_ck, n_rows = status(args.out_root, s, args.tag)
        mark = {"done": "done", "partial": "partial", "pending": "pending"}[st]
        note = f"{n_ck} ckpt"
        if st == "done":
            note += f", {n_rows} fold rows"
        elif st == "partial":
            note += ", no fold_records.csv -> restarts from fold 0"
        print(f"  seed{s:<3}{mark:<10}{note}")
        if st != "done" or args.force:
            todo.append(s)

    if not todo:
        print("\nnothing to do; pass --force to rerun")
        return
    print(f"\nto run: {todo}")
    if args.dry_run:
        for s in todo:
            print(f"  CUDA_VISIBLE_DEVICES=<gpu> python -u {trainer} "
                  f"--tag {args.tag} --seeds {s} "
                  f"--out-dir {seed_dir(args.out_root, s)}")
        return

    q, live = list(todo), {}
    t0 = time.time()
    print(f"\n{len(q)} seed-run(s) over {len(args.gpus)} GPU(s), five folds each\n")
    try:
        while q or live:
            for g in args.gpus:
                if g in live and live[g][0].poll() is not None:
                    p, fh, log, s = live.pop(g)
                    fh.close()
                    st, n_ck, _ = status(args.out_root, s, args.tag)
                    print(f"  done   seed={s} gpu={g} exit={p.returncode} "
                          f"{st} {n_ck} ckpt   [{(time.time()-t0)/60:.0f} min]",
                          flush=True)
                if g not in live and q:
                    s = q.pop(0)
                    p, fh, log = launch(s, g, args.out_root, args.tag,
                                        trainer, args.extra)
                    live[g] = (p, fh, log, s)
                    print(f"  start  seed={s} gpu={g}   {log}", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\ninterrupted; running seeds are left alone. Re-run this script "
              "to pick up\nwhatever did not finish.")
        return

    print(f"\nfinished in {(time.time()-t0)/60:.0f} min\n")
    import pandas as pd
    print(f"{'seed':<7}{'ckpt':>6}{'pooled OOF':>13}{'recorded':>11}"
          f"{'diff':>8}   fold-mean best")
    for s in args.seeds:
        st, n_ck, _ = status(args.out_root, s, args.tag)
        log = Path(args.out_root).parent / f"log_{args.tag}_seed{s}.log"
        got = None
        if log.exists():
            m = re.findall(r"pooled OOF AUROC\s+(\d\.\d+)", log.read_text())
            got = float(m[-1]) if m else None
        fm = ""
        fr = seed_dir(args.out_root, s) / args.tag / "fold_records.csv"
        if fr.exists():
            r = pd.read_csv(fr)
            fm = f"{r.auc_best.mean():.4f} ± {r.auc_best.std(ddof=1):.4f}"
        ref = RECORDED_POOLED.get(s)
        line = f"seed{s:<3}{n_ck:>6}"
        if got is not None:
            line += f"{got:>13.4f}"
            if ref is not None:
                d = abs(got - ref)
                line += f"{ref:>11.3f}{d:>8.3f}"
                line += "  ok" if d <= TOLERANCE else "  <-- CHECK"
            line += f"   {fm}"
        else:
            line += f"{'no pooled OOF in log':>13}"
        print(line)

    print(f"\n  A pooled OOF far from the recorded value means something other "
          f"than GPU\n  nondeterminism changed: library versions, the graph "
          f"files, or the fold\n  assignment. Resolve that before the external "
          f"evaluation.")
    print(f"  Checkpoints carry selected_by=holdout_best with node_mu/node_sd "
          f"and\n  concept_mu/concept_sd, which is everything CPTAC inference "
          f"needs.")


if __name__ == "__main__":
    main()
