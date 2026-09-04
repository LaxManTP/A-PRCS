######################
## Tommy Peer
## A-PRCS
## Meerkat Detection Node
######################

import argparse
import signal
import sys
import time

import cv2 as cv
from ultralytics import YOLO

from pathManager import MODELS_DIR, LOGS_DIR
from riskScorer import RiskScorer


# Default settings
DEFAULT_MODEL      = MODELS_DIR / 'meerkat00.pt'
DEFAULT_CONFIDENCE = 0.25
DEFAULT_FRAME_SKIP = 5      # infer every Nth frame - saves power at the edge
CAMERA_RETRY_MAX   = 30.0   # seconds, exponential backoff ceiling

shutdownRequested = False


def handleSignal(signum, frame):
    global shutdownRequested
    print(f"\n[MEERKAT] Signal {signum} received, shutting down")
    shutdownRequested = True


# Camera

def openCamera(source, width=None, height=None):
    capture = cv.VideoCapture(source)

    if width:
        capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    if not capture.isOpened():
        capture.release()
        return None

    return capture


def reconnectCamera(source, width, height, delay):
    print(f"[MEERKAT] Camera lost - retrying in {delay:.1f}s")
    time.sleep(delay)
    return openCamera(source, width, height)



# Display
def riskColor(score):
    if score >= 0.70:
        return (0, 0, 255)      # red
    if score >= 0.50:
        return (0, 165, 255)    # orange
    if score >= 0.30:
        return (0, 255, 255)    # yellow
    return (0, 255, 0)          # green


def drawOverlay(frame, state):
    color = riskColor(state['score'])

    cv.putText(
        frame,
        f"RISK: {state['score']:.2f} | {state['category']}",
        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
    )
    cv.putText(
        frame,
        f"DETECTIONS: {len(state['detections'])}",
        (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    row = 90

    if state['flags']:
        cv.putText(
            frame, f"FLAGS: {' | '.join(state['flags'])}",
            (10, row), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
        row += 30

    if state['advisory']:
        names = ', '.join(sorted({d['class'] for d in state['advisory']}))
        cv.putText(
            frame, f"UNVERIFIED: {names}",
            (10, row), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
        )
        row += 30

    if state['eventActive']:
        cv.putText(
            frame, f"EVENT {state['eventId']}",
            (10, row), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
        )

    return frame



# Main
def run(args):
    global shutdownRequested

    signal.signal(signal.SIGINT, handleSignal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handleSignal)

    print(f"[MEERKAT] Loading model: {args.model}")
    model = YOLO(str(args.model))
    print(f"[MEERKAT] Classes: {list(model.names.values())}")

    scorer = RiskScorer(nodeId=args.nodeId, logDir=str(args.logDir))

    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = openCamera(source, args.width, args.height)

    if capture is None:
        print(f"[MEERKAT] Camera not available at source {source}, entering retry loop")

    print(f"[MEERKAT] Node {args.nodeId} online | frame skip {args.frameSkip} | "
          f"display {'on' if args.display else 'off'}")

    frameCount = 0
    retryDelay = 1.0
    lastResults = None
    state = {
        'score': 0.0, 'category': 'INITIALIZING', 'flags': [],
        'detections': [], 'advisory': [], 'eventActive': False,
        'eventId': None, 'threatCount': 0, 'wildlifeCount': 0,
        'closedEvent': None,
    }

    try:
        while not shutdownRequested:

            # Frame acquisition

            if capture is None or not capture.isOpened():
                capture = reconnectCamera(source, args.width, args.height, retryDelay)
                retryDelay = min(retryDelay * 2, CAMERA_RETRY_MAX)
                continue

            success, frame = capture.read()
            if not success or frame is None:
                capture.release()
                capture = None
                continue

            retryDelay = 1.0
            frameCount += 1

             # Inference

            ranInference = frameCount % args.frameSkip == 0

            if ranInference:
                try:
                    results = model(frame, conf=args.confidence, verbose=False)
                    lastResults = results

                    detections = []
                    for result in results:
                        for box in result.boxes:
                            detections.append({
                                'class':      model.names[int(box.cls[0])],
                                'confidence': float(box.conf[0]),
                                'box':        box.xyxy[0].tolist(),
                            })

                    state = scorer.process(detections, frameShape=frame.shape)

                    if state['closedEvent']:
                        event = state['closedEvent']
                        print(f"[MEERKAT] EVENT CLOSED {event['eventId']} | "
                              f"peak {event['riskAssessment']['peakScore']:.2f} "
                              f"({event['riskAssessment']['category']}) | "
                              f"{event['durationS']:.0f}s | "
                              f"{', '.join(event['classes'])}")

                except Exception as exc:
                    # One bad frame must not end the deployment.
                    print(f"[MEERKAT] Inference error on frame {frameCount}: {exc}")
                    continue

            ######################
            ## DISPLAY
            ######################

            if args.display:
                if ranInference and lastResults is not None:
                    annotated = lastResults[0].plot()
                else:
                    annotated = frame

                cv.imshow(f'MEERKAT - {args.nodeId}', drawOverlay(annotated, state))

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        # Shutdown
        scorer.shutdown()

        if capture is not None:
            capture.release()
        if args.display:
            cv.destroyAllWindows()

        print("[MEERKAT] System offline")
        print(f"[MEERKAT] Frames processed: {frameCount}")
        print(f"[MEERKAT] Activity log: {scorer.activityLogPath}")
        print(f"[MEERKAT] Threat log:   {scorer.threatLogPath}")


def parseArgs():
    parser = argparse.ArgumentParser(description='A-PRCS Meerkat detection node')

    parser.add_argument('--model',      dest='model',      default=str(DEFAULT_MODEL))
    parser.add_argument('--source',     dest='source',     default='0',
                        help='Camera index, file path, or stream URL')
    parser.add_argument('--node-id',    dest='nodeId',     default='MEERKAT_01')
    parser.add_argument('--log-dir',    dest='logDir',     default=str(LOGS_DIR))
    parser.add_argument('--confidence', dest='confidence', type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument('--frame-skip', dest='frameSkip',  type=int,   default=DEFAULT_FRAME_SKIP)
    parser.add_argument('--width',      dest='width',      type=int,   default=None)
    parser.add_argument('--height',     dest='height',     type=int,   default=None)
    parser.add_argument('--display',    dest='display',    action='store_true',
                        help='Show annotated video window')

    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(run(parseArgs()))