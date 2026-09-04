######################
## Tommy Peer
## A-PRCS
## YOLO Label Class Remapping
######################
##   1. filename.endswith('#####') was a placeholder that never
##      matched anything. Every file was skipped and the script
##      reported success. It is now '.txt'.
##
##   2. Only train/labels was processed. valid/ and test/ kept the
##      old class ids, which means validation metrics were scoring
##      against different labels than training used.
##
##   3. Not idempotent in a dangerous direction. Running 0->4 twice
##      is harmless, but running it against an ALREADY-MERGED
##      directory relabels real class-0 objects as pistols. Now
##      guarded by dry-run-by-default plus pre-flight checks.
##
## TO ADD A NEW DATASET: add an entry to REMAP_JOBS below. That is
## the only part of this file you should need to touch.
######################

import argparse
from collections import Counter
from pathlib import Path

from pathManager import DATASETS_DIR

SPLITS = ('train', 'valid', 'test', 'val')

######################
## MERGED CLASS LIST
######################
##
## Order here IS the class id order and must match names: in
## fullSet.yaml exactly. Add to the END - inserting in the middle
## renumbers everything downstream and silently invalidates every
## label file you have already written.
######################

APRCS_CLASSES = [
    'buffalo',    # 0
    'elephant',   # 1
    'rhino',      # 2
    'zebra',      # 3
    'pistol',     # 4
    # 'person',   # 5
    # 'rifle',    # 6
    # 'drone',    # 7
    # 'vehicle',  # 8
]

######################
## REMAP JOBS
######################
##
## One entry per source dataset. Edit as each new dataset arrives.
##
##   name     short label used by --only and in output
##   root     dataset root containing train/ valid/ test/
##   mapping  {sourceClassId: aprcsClassId}. Map to None to DROP
##            that class - useful when a downloaded set ships with
##            classes you do not want (knives, backgrounds, etc).
##   expect   class ids you expect on disk BEFORE remapping. The
##            run aborts if what is there does not match, which is
##            the guard against running twice.
##   enabled  set False once applied so a bulk run skips it.
##
## Mapping is applied in a single pass per line, so 0->4 and 4->5
## in the same job do NOT cascade.
######################

REMAP_JOBS = [
    {
        'name':    'pistols',
        'root':    DATASETS_DIR / 'fullSet' / 'Pistols.v1-resize-416x416.yolo26',
        'mapping': {0: 4},
        'expect':  {0},
        'enabled': False,   # already applied - matrix confirms pistol is class 4
    },

    # Template for the next dataset. Fill in and flip enabled to True.
    # {
    #     'name':    'person',
    #     'root':    DATASETS_DIR / 'fullSet' / 'personDataset',
    #     'mapping': {0: 5, 1: None},   # keep person as 5, drop class 1
    #     'expect':  {0, 1},
    #     'enabled': True,
    # },
]


######################
## SCANNING
######################

def scanClasses(labelsDir):
    """Count class id occurrences without modifying anything."""
    counts = Counter()

    for filepath in Path(labelsDir).glob('*.txt'):
        for line in filepath.read_text().splitlines():
            parts = line.split()
            if parts:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    print(f"    [WARN] Non-numeric class id in {filepath.name}: {line!r}")

    return counts


######################
## REMAPPING
######################

def remapDirectory(labelsDir, mapping, dryRun=True):
    labelsDir = Path(labelsDir)
    if not labelsDir.is_dir():
        print(f"[SKIP] Not a directory: {labelsDir}")
        return 0, 0, 0

    changedFiles = 0
    untouchedFiles = 0
    changedLines = 0
    droppedLines = 0

    for filepath in sorted(labelsDir.glob('*.txt')):
        lines = filepath.read_text().splitlines()
        newLines = []
        fileChanged = False

        for line in lines:
            parts = line.split()
            if not parts:
                newLines.append(line)
                continue

            try:
                classId = int(parts[0])
            except ValueError:
                # Preserve anything unparseable rather than dropping it
                newLines.append(line)
                continue

            if classId in mapping:
                target = mapping[classId]

                if target is None:
                    # Drop this annotation entirely
                    fileChanged = True
                    droppedLines += 1
                    continue

                parts[0] = str(target)
                fileChanged = True
                changedLines += 1

            newLines.append(' '.join(parts))

        if fileChanged:
            changedFiles += 1
            if not dryRun:
                # An emptied label file is valid YOLO - it means
                # "this image is pure background". Keep it.
                body = '\n'.join(newLines)
                filepath.write_text(body + '\n' if body else '')
        else:
            untouchedFiles += 1

    verb = 'Would change' if dryRun else 'Changed'
    print(f"    {verb}: {changedFiles} files | {changedLines} remapped, "
          f"{droppedLines} dropped | untouched: {untouchedFiles}")

    return changedFiles, changedLines, droppedLines


def remapDataset(datasetRoot, mapping, dryRun=True, expectClasses=None, force=False):
    """
    Walks every split under datasetRoot and applies mapping.
    Returns True if the remap ran (or would run), False if aborted.
    """
    datasetRoot = Path(datasetRoot)
    if not datasetRoot.is_dir():
        print(f"  [ABORT] Dataset root not found: {datasetRoot}")
        return False

    labelDirs = [
        datasetRoot / split / 'labels'
        for split in SPLITS
        if (datasetRoot / split / 'labels').is_dir()
    ]

    if not labelDirs:
        print(f"  [ABORT] No <split>/labels directories under {datasetRoot}")
        print(f"          Looked for: {', '.join(SPLITS)}")
        return False

    ######################
    ## PRE-FLIGHT
    ######################

    print("  Pre-flight scan:")

    allFound = Counter()
    for labelsDir in labelDirs:
        counts = scanClasses(labelsDir)
        allFound.update(counts)
        summary = ', '.join(f"{k}:{v}" for k, v in sorted(counts.items())) or 'empty'
        print(f"    {labelsDir.parent.name:<6} {summary}")

    present = set(allFound)
    targets = {v for v in mapping.values() if v is not None}

    if not force:
        alreadyPresent = targets & present
        if alreadyPresent:
            print(f"\n  [ABORT] Target class ids {sorted(alreadyPresent)} already on disk.")
            print(f"          This dataset may already be remapped. Re-running would")
            print(f"          corrupt genuine annotations. Override with --force.")
            return False

        if expectClasses is not None and present != set(expectClasses):
            print(f"\n  [ABORT] Expected {sorted(expectClasses)}, found {sorted(present)}.")
            print(f"          Refusing to modify a dataset that does not look like")
            print(f"          the one this job was written for.")
            return False

    sources = set(mapping)
    if not (sources & present):
        print(f"\n  [ABORT] None of the source ids {sorted(sources)} are present.")
        return False

    ######################
    ## REMAP
    ######################

    readable = ', '.join(
        f"{src}->{'DROP' if dst is None else dst}"
        for src, dst in sorted(mapping.items())
    )
    print(f"\n  Applying: {readable}")

    for labelsDir in labelDirs:
        print(f"    {labelsDir.parent.name}/")
        remapDirectory(labelsDir, mapping, dryRun)

    if not dryRun:
        print("\n  Post-flight verify:")
        for labelsDir in labelDirs:
            counts = scanClasses(labelsDir)
            summary = ', '.join(f"{k}:{v}" for k, v in sorted(counts.items())) or 'empty'
            print(f"    {labelsDir.parent.name:<6} {summary}")

    return True


######################
## JOB RUNNER
######################

def runJobs(jobs=None, dryRun=True, force=False, only=None):
    """Runs every enabled job in REMAP_JOBS."""
    jobs = jobs if jobs is not None else REMAP_JOBS

    print("######################")
    print(f"## A-PRCS LABEL REMAP  [{'DRY RUN' if dryRun else 'WRITING'}]")
    print("######################")
    print(f"Merged classes: {list(enumerate(APRCS_CLASSES))}\n")

    ran, skipped, failed = 0, 0, 0

    for job in jobs:
        name = job['name']

        if only and name not in only:
            continue

        if not job.get('enabled', True) and not only:
            print(f"[SKIP] {name} (disabled)")
            skipped += 1
            continue

        print(f"\n--- {name} ---")
        print(f"  {job['root']}")

        ok = remapDataset(
            datasetRoot   = job['root'],
            mapping       = job['mapping'],
            dryRun        = dryRun,
            expectClasses = job.get('expect'),
            force         = force,
        )

        if ok:
            ran += 1
        else:
            failed += 1

    print(f"\n######################")
    print(f"## {ran} ran, {skipped} skipped, {failed} aborted")
    if dryRun and ran:
        print("## Nothing written. Re-run with --apply to commit.")
    print("######################")

    return ran, skipped, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remap YOLO label class ids across all dataset splits. '
                    'Edit REMAP_JOBS at the top of this file to add datasets.'
    )
    parser.add_argument('--apply', action='store_true',
                        help='Actually write (default is dry run)')
    parser.add_argument('--force', action='store_true',
                        help='Skip the already-remapped safety guards')
    parser.add_argument('--only',  nargs='*', default=None,
                        help='Run only these job names, ignoring enabled flags')
    args = parser.parse_args()

    if args.force:
        print("[WARN] --force: safety guards disabled\n")

    runJobs(dryRun=not args.apply, force=args.force, only=args.only)