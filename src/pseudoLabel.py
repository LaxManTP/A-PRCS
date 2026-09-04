######################
## Tommy Peer
## A-PRCS
## Pseudo-Labeling for Partial-Label Repair
######################
##
## THE PROBLEM THIS SOLVES
##
## The pistol dataset contains people who are not annotated. When
## YOLO trains on a merged set, every unlabeled human becomes an
## explicit negative: "this region is NOT a person." The model
## learns an anti-correlation - weapon visible, suppress person -
## so person recall is worst in exactly the scenes that matter most.
## Since ARMED_PERSON in riskScorer.py requires BOTH classes, the
## risk score collapses at the worst possible moment.
##
## Same bug runs the other direction: the wildlife set almost
## certainly contains unlabeled humans (researchers, tourists, the
## person servicing the camera). Add a job for it.
##
## THE FIX
##
## Run a COCO-pretrained detector over the affected images, accept
## high-confidence boxes for the missing class, merge into existing
## labels. This is a repair pass, not a labeling strategy - review
## the output before trusting it.
##
## TO ADD A DATASET: add an entry to PSEUDO_LABEL_JOBS below.
######################

import argparse
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from pathManager import DATASETS_DIR, PROJECT_ROOT

SPLITS = ('train', 'valid', 'test', 'val')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Pseudo label jobs
##
##   name        short label used by --only and in output
##   root        dataset root containing train/ valid/ test/
##   model       COCO-pretrained weights (NOT your meerkat model -
##               it does not know 'person' yet, that is the point)
##   accept      {cocoClassName: aprcsClassId} to write
##   minConf     confidence floor. Start high. Run --scan first.
##   iouSkip     skip a pseudo-box overlapping an existing box by
##               more than this, to avoid double-labeling
##   minBoxArea  drop boxes smaller than this fraction of the frame,
##               which filters spurious background detections
##   enabled     set False once applied so a bulk run skips it


PSEUDO_LABEL_JOBS = [
    {
        'name':       'pistolsPerson',
        'root':       DATASETS_DIR / 'fullSet' / 'Pistols.v1-resize-416x416.yolo26',
        'model':      'yolo26n.pt',
        'accept':     {'person': 5},
        'minConf':    0.50,
        'iouSkip':    0.45,
        'minBoxArea': 0.002,
        'enabled':    True,
    },

    # The wildlife set has the same problem in reverse. Enable once
    # 'person' is confirmed as class 5 in fullSet.yaml.
    # {
    #     'name':       'wildlifePerson',
    #     'root':       DATASETS_DIR / 'fullSet' / 'africanWildlife',
    #     'model':      'yolo26n.pt',
    #     'accept':     {'person': 5},
    #     'minConf':    0.55,
    #     'iouSkip':    0.45,
    #     'minBoxArea': 0.002,
    #     'enabled':    False,
    # },
]

REVIEW_DIR = PROJECT_ROOT / 'runs' / 'pseudoLabel'
REVIEW_LIMIT = 60   # annotated images written per job for eyeballing


# Box geometry
def xyxyToYolo(box, imageWidth, imageHeight):
    """Absolute [x1,y1,x2,y2] -> normalized [xc,yc,w,h]."""
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / imageWidth
    yc = ((y1 + y2) / 2.0) / imageHeight
    w  = (x2 - x1) / imageWidth
    h  = (y2 - y1) / imageHeight
    return [
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    ]


def yoloToXyxy(box):
    """Normalized [xc,yc,w,h] -> normalized [x1,y1,x2,y2]."""
    xc, yc, w, h = box
    return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]


def boxIou(boxA, boxB):
    """IoU between two normalized YOLO boxes."""
    ax1, ay1, ax2, ay2 = yoloToXyxy(boxA)
    bx1, by1, bx2, by2 = yoloToXyxy(boxB)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    interW = max(ix2 - ix1, 0.0)
    interH = max(iy2 - iy1, 0.0)
    intersection = interW * interH

    if intersection <= 0:
        return 0.0

    areaA = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    areaB = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = areaA + areaB - intersection

    return intersection / union if union > 0 else 0.0



# Label file io
def readLabels(labelPath):
    """Returns list of (classId, [xc,yc,w,h]). Missing file -> empty."""
    labelPath = Path(labelPath)
    if not labelPath.exists():
        return []

    entries = []
    for line in labelPath.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            entries.append((int(parts[0]), [float(v) for v in parts[1:5]]))
        except ValueError:
            continue

    return entries


def writeLabels(labelPath, entries):
    lines = [
        f"{classId} " + ' '.join(f"{v:.6f}" for v in box)
        for classId, box in entries
    ]
    body = '\n'.join(lines)
    Path(labelPath).write_text(body + '\n' if body else '')


def findImages(imagesDir):
    imagesDir = Path(imagesDir)
    if not imagesDir.is_dir():
        return []
    return sorted(
        p for p in imagesDir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )


######################
## PREDICTION
######################

def predictPseudoBoxes(model, imagePath, accept, minConf, minBoxArea):
    """
    Runs the COCO model on one image. Returns list of
    (aprcsClassId, [xc,yc,w,h], confidence) for accepted classes only.
    """
    results = model(str(imagePath), conf=minConf, verbose=False)
    proposals = []

    for result in results:
        height, width = result.orig_shape

        for box in result.boxes:
            cocoName = model.names[int(box.cls[0])]
            if cocoName not in accept:
                continue

            confidence = float(box.conf[0])
            yoloBox = xyxyToYolo(box.xyxy[0].tolist(), width, height)

            if yoloBox[2] * yoloBox[3] < minBoxArea:
                continue

            proposals.append((accept[cocoName], yoloBox, confidence))

    return proposals


def mergeProposals(existing, proposals, iouSkip):
    """
    Adds proposals that do not overlap an existing box of the SAME
    class. Overlap with a different class is fine - a person holding
    a pistol produces two legitimately overlapping boxes.

    Returns (mergedEntries, addedCount, skippedCount).
    """
    merged = list(existing)
    added = 0
    skipped = 0

    for classId, box, _confidence in proposals:
        duplicate = any(
            existingClass == classId and boxIou(existingBox, box) > iouSkip
            for existingClass, existingBox in merged
        )

        if duplicate:
            skipped += 1
            continue

        merged.append((classId, box))
        added += 1

    return merged, added, skipped



# Review Images
def writeReviewImage(imagePath, existing, proposals, outputPath, classNames=None):
    """
    Existing boxes green, pseudo-boxes magenta with confidence.
    Import is local so the module still loads without cv2 present.
    """
    import cv2 as cv

    image = cv.imread(str(imagePath))
    if image is None:
        return False

    height, width = image.shape[:2]

    def toPixels(box):
        x1, y1, x2, y2 = yoloToXyxy(box)
        return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    for classId, box in existing:
        x1, y1, x2, y2 = toPixels(box)
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = classNames[classId] if classNames and classId < len(classNames) else str(classId)
        cv.putText(image, label, (x1, max(y1 - 6, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for classId, box, confidence in proposals:
        x1, y1, x2, y2 = toPixels(box)
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = classNames[classId] if classNames and classId < len(classNames) else str(classId)
        cv.putText(image, f"{label} {confidence:.2f}", (x1, max(y1 - 6, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    outputPath = Path(outputPath)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(outputPath), image)
    return True


# Split processing
def backupLabels(labelsDir):
    labelsDir = Path(labelsDir)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = labelsDir.parent / f'labels_backup_{stamp}'
    shutil.copytree(labelsDir, backup)
    print(f"    Backup: {backup}")
    return backup


def processSplit(splitDir, model, job, dryRun, scanOnly, reviewDir, reviewBudget):
    imagesDir = splitDir / 'images'
    labelsDir = splitDir / 'labels'

    images = findImages(imagesDir)
    if not images:
        print(f"    No images in {imagesDir}")
        return Counter(), []

    stats = Counter()
    confidences = []
    pending = []

    for imagePath in images:
        labelPath = labelsDir / f'{imagePath.stem}.txt'
        existing = readLabels(labelPath)

        proposals = predictPseudoBoxes(
            model, imagePath,
            job['accept'], job['minConf'], job['minBoxArea'],
        )

        stats['images'] += 1

        if not proposals:
            stats['imagesUnchanged'] += 1
            continue

        confidences.extend(c for _, _, c in proposals)

        merged, added, skipped = mergeProposals(existing, proposals, job['iouSkip'])

        stats['proposed'] += len(proposals)
        stats['added'] += added
        stats['skippedOverlap'] += skipped

        if added:
            stats['imagesChanged'] += 1
            if not labelPath.exists():
                stats['labelFilesCreated'] += 1
            pending.append((labelPath, merged, imagePath, existing, proposals))

 # Review and write
    if reviewDir is not None:
        written = 0
        for labelPath, _merged, imagePath, existing, proposals in pending:
            if written >= reviewBudget:
                break
            ok = writeReviewImage(
                imagePath, existing, proposals,
                reviewDir / splitDir.name / imagePath.name,
                classNames=APRCS_CLASS_NAMES,
            )
            written += 1 if ok else 0
        if written:
            print(f"    Review images: {written} -> {reviewDir / splitDir.name}")

    if not dryRun and not scanOnly and pending:
        backupLabels(labelsDir)
        for labelPath, merged, _imagePath, _existing, _proposals in pending:
            writeLabels(labelPath, merged)

    return stats, confidences


# Job runner

APRCS_CLASS_NAMES = [
    'buffalo', 'elephant', 'rhino', 'zebra', 'pistol',
    'person', 'rifle', 'drone', 'vehicle',
]


def reportConfidences(confidences):
    if not confidences:
        print("    No detections above threshold.")
        return

    confidences = sorted(confidences)
    buckets = Counter()
    for c in confidences:
        buckets[f"{int(c * 10) / 10:.1f}"] += 1

    print("    Confidence distribution:")
    for edge in sorted(buckets):
        count = buckets[edge]
        bar = '#' * min(count // max(len(confidences) // 40, 1), 40)
        print(f"      {edge}-{float(edge) + 0.1:.1f}  {count:>5}  {bar}")

    mid = len(confidences) // 2
    print(f"    min {confidences[0]:.2f} | median {confidences[mid]:.2f} | "
          f"max {confidences[-1]:.2f} | n={len(confidences)}")


def runJob(job, dryRun=True, scanOnly=False, review=True):
    from ultralytics import YOLO

    root = Path(job['root'])
    if not root.is_dir():
        print(f"  [ABORT] Dataset root not found: {root}")
        return False

    splitDirs = [
        root / split for split in SPLITS
        if (root / split / 'images').is_dir()
    ]

    if not splitDirs:
        print(f"  [ABORT] No <split>/images directories under {root}")
        return False

    print(f"  Model: {job['model']}")
    model = YOLO(job['model'])

    unknown = [name for name in job['accept'] if name not in model.names.values()]
    if unknown:
        print(f"  [ABORT] Model does not know these classes: {unknown}")
        print(f"          Use a COCO-pretrained checkpoint, not a meerkat model.")
        return False

    mapping = ', '.join(
        f"{name}->{APRCS_CLASS_NAMES[cid] if cid < len(APRCS_CLASS_NAMES) else cid} ({cid})"
        for name, cid in job['accept'].items()
    )
    print(f"  Accepting: {mapping} @ conf>={job['minConf']}")

    reviewDir = (REVIEW_DIR / job['name']) if review else None
    totals = Counter()
    allConfidences = []

    for splitDir in splitDirs:
        print(f"    {splitDir.name}/")
        stats, confidences = processSplit(
            splitDir, model, job, dryRun, scanOnly, reviewDir, REVIEW_LIMIT,
        )
        totals.update(stats)
        allConfidences.extend(confidences)

        print(f"      {stats['images']} images | {stats['imagesChanged']} would gain boxes | "
              f"{stats['added']} added, {stats['skippedOverlap']} skipped as duplicate")

    print(f"\n  TOTAL: {totals['images']} images scanned")
    print(f"         {totals['imagesChanged']} images affected "
          f"({100 * totals['imagesChanged'] / max(totals['images'], 1):.1f}%)")
    print(f"         {totals['added']} boxes added, "
          f"{totals['skippedOverlap']} skipped as overlapping existing")
    if totals['labelFilesCreated']:
        print(f"         {totals['labelFilesCreated']} label files created from scratch")

    reportConfidences(allConfidences)
    return True


def runJobs(jobs=None, dryRun=True, scanOnly=False, review=True, only=None):
    jobs = jobs if jobs is not None else PSEUDO_LABEL_JOBS

    mode = 'SCAN ONLY' if scanOnly else ('DRY RUN' if dryRun else 'WRITING')
    print("######################")
    print(f"## A-PRCS PSEUDO-LABEL  [{mode}]")
    print("######################")

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

        if runJob(job, dryRun=dryRun, scanOnly=scanOnly, review=review):
            ran += 1
        else:
            failed += 1

    print(f"\n######################")
    print(f"## {ran} ran, {skipped} skipped, {failed} aborted")
    if (dryRun or scanOnly) and ran:
        print("## Nothing written.")
        print("## Check the review images, then re-run with --apply.")
    print("######################")

    return ran, skipped, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pseudo-label missing classes in partially-labeled datasets. '
                    'Edit PSEUDO_LABEL_JOBS at the top of this file.'
    )
    parser.add_argument('--apply',     action='store_true',
                        help='Actually write labels (default is dry run)')
    parser.add_argument('--scan',      action='store_true',
                        help='Report confidence distribution only, no review images')
    parser.add_argument('--no-review', dest='noReview', action='store_true',
                        help='Skip writing annotated review images')
    parser.add_argument('--only',      nargs='*', default=None,
                        help='Run only these job names, ignoring enabled flags')
    args = parser.parse_args()

    runJobs(
        dryRun   = not args.apply,
        scanOnly = args.scan,
        review   = not (args.noReview or args.scan),
        only     = args.only,
    )