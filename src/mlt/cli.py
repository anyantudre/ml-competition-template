"""Command line: `mlt <command> --help` for details."""

from __future__ import annotations

import argparse
import sys

from .utils import LOG, setup_logging


def _cfg(args):
    from .config import load_config

    data = {"name": args.name} if getattr(args, "name", None) else None
    return load_config(args.config, args.set, data=data)


def cmd_init(args):
    from .scaffold import scaffold

    scaffold(args.train, args.test, args.sample, args.out, metric=args.metric, name=args.name)


def cmd_example(args):
    from .examples import make_example

    make_example(args.dataset, args.out)


def cmd_folds(args):
    from .folds import default_scheme, describe
    from .train import prepare

    cfg = _cfg(args)
    ctx = prepare(cfg, force_folds=args.force)
    print(f"folds file: {ctx['folds_path']}  (scheme: {default_scheme(cfg, ctx['task'])})")
    if ctx["task"] != "segmentation":
        print(describe(ctx["folds"], ctx["y"], ctx["task"]).to_string(index=False))


def cmd_run(args):
    from .train import run_cv

    cfg = _cfg(args)
    res = run_cv(cfg, save=True)
    print(f"\n{cfg['name']}: CV {res.score:.5f}  ->  {res.run_dir}")


def cmd_predict(args):
    from .predict import predict_file

    predict_file(args.run, args.input, args.out, args.sample)


def cmd_validate(args):
    from .data import read_table
    from .submission import validate_submission

    sub = read_table(args.submission)
    sample = read_table(args.sample) if args.sample else None
    problems = validate_submission(sub, sample, args.id_col, allow_empty=args.allow_empty)
    if problems:
        for p in problems:
            print(f"problem: {p}")
        sys.exit(1)
    print(f"{args.submission}: ok ({len(sub)} rows)")


def cmd_score(args):
    from .data import read_table
    from .submission import score_submission

    sub, sol = read_table(args.submission), read_table(args.solution)
    id_col = args.id_col or sol.columns[0]
    value = score_submission(sub, sol, args.metric, args.task, id_col)
    print(f"{args.metric}: {value:.5f}")


def cmd_ensemble(args):
    from .ensemble import ensemble_runs

    ensemble_runs(args.runs, method=args.method, name=args.name, output_dir=args.output_dir,
                  stack_model=args.stack_model)


def cmd_compare(args):
    from .compare import compare_runs

    compare_runs(args.run_a, args.run_b, n_boot=args.n_boot, seed=args.seed)


def cmd_tune(args):
    from .tune import tune

    tune(_cfg(args), n_trials=args.trials, timeout=args.timeout)


def cmd_select(args):
    from .tune import forward_select

    forward_select(_cfg(args), max_features=args.max_features)


def cmd_check(args):
    from .checks import run_checks

    run_checks(_cfg(args))


def cmd_serve(args):
    from .serve import serve

    serve(args.run, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlt", description="ML competition toolkit")
    p.add_argument("--debug", action="store_true", help="show full tracebacks")
    sub =p.add_subparsers(dest="command", required=True)

    def with_config(sp):
        sp.add_argument("config", help="experiment YAML")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set model.name=xgb (repeatable)")
        sp.add_argument("--name", help="run name (default: config file name)")
        return sp

    sp = sub.add_parser("init", help="write a starter config for a new competition")
    sp.add_argument("--train", required=True)
    sp.add_argument("--test")
    sp.add_argument("--sample", help="sample_submission file")
    sp.add_argument("--metric")
    sp.add_argument("--name")
    sp.add_argument("--out", help="config path (default: configs/<name>.yaml)")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("example", help="download a small public dataset in competition format")
    sp.add_argument("dataset", help="name, or 'list'")
    sp.add_argument("--out", help="folder (default: data/<dataset>)")
    sp.set_defaults(func=cmd_example)

    sp = with_config(sub.add_parser("folds", help="create (or show) the CV folds file"))
    sp.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    sp.set_defaults(func=cmd_folds)

    sp = with_config(sub.add_parser("run", help="folds + train every fold + OOF + test predictions + submission"))
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("predict", help="predict a new file with a finished run")
    sp.add_argument("run", help="run folder, e.g. outputs/my_run")
    sp.add_argument("--input", required=True)
    sp.add_argument("--sample", help="sample_submission for column order and checks")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("validate", help="check a submission against sample_submission")
    sp.add_argument("submission")
    sp.add_argument("--sample")
    sp.add_argument("--id-col")
    sp.add_argument("--allow-empty", action="store_true", help="empty cells are valid (e.g. RLE)")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("score", help="score a submission against a local solution file")
    sp.add_argument("submission")
    sp.add_argument("--solution", required=True)
    sp.add_argument("--metric", required=True)
    sp.add_argument("--task", required=True)
    sp.add_argument("--id-col")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("ensemble", help="blend runs: mean, rank, weighted or stack")
    sp.add_argument("runs", nargs="+")
    sp.add_argument("--method", default="weighted", choices=["mean", "rank", "weighted", "stack"])
    sp.add_argument("--name", default=None)
    sp.add_argument("--output-dir", default="outputs")
    sp.add_argument("--stack-model", default=None, help="level-2 model for --method stack")
    sp.set_defaults(func=cmd_ensemble)

    sp = sub.add_parser("compare", help="paired bootstrap on OOF predictions: is B really better?")
    sp.add_argument("run_a")
    sp.add_argument("run_b")
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_compare)

    sp = with_config(sub.add_parser("tune", help="Optuna search over tune.space (or a default space)"))
    sp.add_argument("--trials", type=int)
    sp.add_argument("--timeout", type=int, help="seconds")
    sp.set_defaults(func=cmd_tune)

    sp = with_config(sub.add_parser("select", help="greedy forward feature selection on CV score"))
    sp.add_argument("--max-features", type=int)
    sp.set_defaults(func=cmd_select)

    sp = with_config(sub.add_parser("check", help="data checks, leakage hints and adversarial validation"))
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("serve", help="HTTP API around a finished run")
    sp.add_argument("run")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv=None):
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyError, ValueError, FileNotFoundError, ImportError) as exc:
        if args.debug:
            raise
        LOG.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
