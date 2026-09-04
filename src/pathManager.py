######################
## Tommy Peer
## A-PRCS
## Canonical Project Paths
######################

from pathlib import Path

# src/paths.py -> src -> aprcs
# If you move this file, fix the number of .parent calls.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR      = PROJECT_ROOT / 'src'
MODELS_DIR   = PROJECT_ROOT / 'models'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RUNS_DIR     = PROJECT_ROOT / 'runs'
LOGS_DIR     = PROJECT_ROOT / 'logs'
DOCS_DIR     = PROJECT_ROOT / 'docs'

DATA_YAML    = DATASETS_DIR / 'fullSet' / 'fullSet.yaml'

# Directories that must exist at runtime
for _d in (MODELS_DIR, RUNS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def alignUltralyticsSettings():

    # To be called once at the beginning of training sessions
    from ultralytics import settings

    desired = {
        'runs_dir':     str(RUNS_DIR),
        'datasets_dir': str(DATASETS_DIR),
    }

    changed = {k: v for k, v in desired.items() if settings.get(k) != v}

    if changed:
        for key, value in changed.items():
            print(f"[PATHS] Ultralytics {key}: {settings.get(key)} -> {value}")
        settings.update(changed)
    else:
        print("[PATHS] Ultralytics settings already aligned")

    return settings


def describe():
    print("######################")
    print("## A-PRCS PATHS")
    print("######################")
    for name, path in [
        ('PROJECT_ROOT', PROJECT_ROOT),
        ('MODELS_DIR',   MODELS_DIR),
        ('DATASETS_DIR', DATASETS_DIR),
        ('RUNS_DIR',     RUNS_DIR),
        ('LOGS_DIR',     LOGS_DIR),
        ('DATA_YAML',    DATA_YAML),
    ]:
        exists = 'OK ' if Path(path).exists() else 'MISSING'
        print(f"  [{exists}] {name:<13} {path}")


if __name__ == '__main__':
    describe()