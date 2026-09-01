######################
## Tommy Peer
## A-PRCS
## Model Training
######################

import argparse
import os
from datetime import datetime

# Must be set before torch initializes CUDA. Lets the allocator grow
# segments instead of demanding one contiguous block, which is what
# fails after fragmentation builds up over many epochs.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from ultralytics import YOLO

from pathManager import (
    PROJECT_ROOT, RUNS_DIR, MODELS_DIR, DATA_YAML,
    alignUltralyticsSettings, describe,
)

######################
## TRAINING CONFIG
######################
DEFAULTS = {
    'epochs':   150,
    'imgsz':    640,
    'batch':    8,
    'patience': 30,
    'device':   '0',
}


def meerkatTraining(name=None, resume=False, **overrides):
    cfg = {**DEFAULTS, **overrides}

    # Force Ultralytics' persistent settings to point at THIS project
    # before anything resolves a path.
    alignUltralyticsSettings()
    describe()

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"[TRAINING] Dataset config not found: {DATA_YAML}\n"
            f"            Check DATASETS_DIR in paths.py"
        )

    # Timestamped run name so runs never silently collide or auto-increment
    # into meerkat002, meerkat003, ... with no record of which is which.
    if name is None:
        name = f"meerkat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"[TRAINING] torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[TRAINING] Device: {torch.cuda.get_device_name(0)}")
    else:
        print("[TRAINING] WARNING - no CUDA device, training on CPU will be very slow")
        cfg['device'] = 'cpu'

    print(f"[TRAINING] Run name: {name}")
    print(f"[TRAINING] Output:   {RUNS_DIR / 'training' / name}")

    if resume:
        checkpoint = findLastCheckpoint(name if name else None)
        if checkpoint is None:
            raise FileNotFoundError(
                f"[TRAINING] No last.pt found under {RUNS_DIR / 'training'}"
            )
        print(f"[TRAINING] Resuming from {checkpoint}")
        print(f"[TRAINING] NOTE - batch/imgsz come from the checkpoint, "
              f"not from DEFAULTS")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return model, model.val(split='test', imgsz=cfg['imgsz'], plots=True)

    model = YOLO('yolo26n.pt')

    model.train(
        data     = str(DATA_YAML),
        epochs   = cfg['epochs'],
        imgsz    = cfg['imgsz'],
        batch    = cfg['batch'],
        patience = cfg['patience'],
        device   = cfg['device'],

        # ABSOLUTE path. A relative 'runs/training' resolves against the
        # current working directory, which is why output was landing in
        # a previous project folder.
        project  = str(RUNS_DIR / 'training'),
        name     = name,
        exist_ok = False,
        resume   = resume,

        ######################
        ## SMALL-OBJECT / DOMAIN-GAP TUNING
        ######################

        # The pistol set and the wildlife set are different visual
        # domains. Mosaic helps force the model to key on the object
        # rather than the background, but late-stage mosaic hurts
        # localization, so close it for the final 15 epochs.
        mosaic          = 1.0,
        close_mosaic    = 15,

        # Scale jitter widens the apparent size range of small objects.
        scale           = 0.5,

        # Colour jitter reduces reliance on scene-level colour cues
        # (savanna warm tones vs indoor pistol imagery).
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,

        # Wildlife cameras see animals and people in both orientations.
        fliplr          = 0.5,
        flipud          = 0.0,

        # Cache images in RAM if you have the headroom - big speedup.
        cache           = False,

        # Keep plots so the confusion matrix regenerates every run.
        plots           = True,
        val             = True,
    )

    ######################
    ## EVALUATION
    ######################
    ##
    ## Evaluate on the held-out test split, not val. model.val() with
    ## no args re-runs the validation split the model already tuned
    ## early stopping against, which flatters the numbers.
    ######################

    print("[TRAINING] Evaluating on test split...")
    metrics = model.val(split='test', imgsz=cfg['imgsz'], plots=True)

    print("[TRAINING] Per-class results:")
    try:
        for idx, className in model.names.items():
            precision, recall, ap50, ap = metrics.class_result(idx)
            print(f"  {className:<10} P={precision:.3f}  R={recall:.3f}  "
                  f"mAP50={ap50:.3f}  mAP50-95={ap:.3f}")
    except Exception as exc:
        print(f"  [could not read per-class metrics: {exc}]")

    # Copy best weights somewhere stable so meerkat.py has a fixed target.
    best = RUNS_DIR / 'training' / name / 'weights' / 'best.pt'
    if best.exists():
        import shutil
        destination = MODELS_DIR / f'{name}.pt'
        shutil.copy2(best, destination)
        print(f"[TRAINING] Best weights copied to {destination}")

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A-PRCS Meerkat model training')
    parser.add_argument('--name',     default=None, help='Run name (default: timestamped)')
    parser.add_argument('--epochs',   type=int,   default=DEFAULTS['epochs'])
    parser.add_argument('--imgsz',    type=int,   default=DEFAULTS['imgsz'])
    parser.add_argument('--batch',    type=int,   default=DEFAULTS['batch'])
    parser.add_argument('--device',   default=DEFAULTS['device'])
    parser.add_argument('--resume',   action='store_true')
    args = parser.parse_args()


    def findLastCheckpoint(runName=None):
        """
        Returns the most recent last.pt under runs/training, or the one
        for a named run. Ultralytics resumes from the checkpoint itself,
        not from the base weights.
        """
        trainingDir = RUNS_DIR / 'training'
        if not trainingDir.is_dir():
            return None

        if runName:
            candidate = trainingDir / runName / 'weights' / 'last.pt'
            return candidate if candidate.exists() else None

        checkpoints = sorted(
            trainingDir.glob('*/weights/last.pt'),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None

    meerkatTraining(
        name   = args.name,
        resume = args.resume,
        epochs = args.epochs,
        imgsz  = args.imgsz,
        batch  = args.batch,
        device = args.device,
    )
if __name__ == '__main__':
    meerkatTraining()