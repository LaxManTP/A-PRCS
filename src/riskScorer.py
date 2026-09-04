######################
## Tommy Peer
## A-PRCS
## Meerkat Risk Scorer
######################

import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone

SCHEMA_VERSION = '2.0.0'


# Class configurations
CLASS_CONFIG = {
    'buffalo':  {'category': 'wildlife', 'baseThreat': 0.05, 'minConf': 0.35},
    'elephant': {'category': 'wildlife', 'baseThreat': 0.05, 'minConf': 0.35},
    'rhino':    {'category': 'wildlife', 'baseThreat': 0.05, 'minConf': 0.35},
    'zebra':    {'category': 'wildlife', 'baseThreat': 0.05, 'minConf': 0.35},

    # Weapons carry a deliberately low gate. A missed pistol costs more
    # than a false alarm, and current model recall on pistol is ~0.64.
    'pistol':   {'category': 'threat',   'baseThreat': 0.90, 'minConf': 0.25},

    # Uncomment as each class enters the model. Note that until 'person'
    # exists, none of COMBINATION_FLAGS below can fire and this scorer
    # reduces to "pistol high, animal low".
    'person':   {'category': 'threat',   'baseThreat': 0.55, 'minConf': 0.40},
    # 'drone':    {'category': 'threat',   'baseThreat': 0.85, 'minConf': 0.35},
    # 'rifle':    {'category': 'threat',   'baseThreat': 0.95, 'minConf': 0.25},
    # 'vehicle':  {'category': 'context',  'baseThreat': 0.30, 'minConf': 0.40},
}

# Detections between (minConf - ADVISORY_BAND) and minConf are surfaced
# as UNVERIFIED rather than silently dropped.
ADVISORY_BAND = 0.15


# Combination flags
COMBINATION_FLAGS = [
    {'classes': frozenset(['person', 'rifle']),    'multiplier': 2.0, 'flag': 'ARMED_PERSON_RIFLE',   'requiresProximity': True,  'maxDistance': 0.30},
    {'classes': frozenset(['person', 'pistol']),   'multiplier': 1.9, 'flag': 'ARMED_PERSON',         'requiresProximity': True,  'maxDistance': 0.30},
    {'classes': frozenset(['person', 'drone']),    'multiplier': 1.5, 'flag': 'COORDINATED_ACTIVITY', 'requiresProximity': False, 'maxDistance': None},
    {'classes': frozenset(['person', 'vehicle']),  'multiplier': 1.3, 'flag': 'VEHICLE_INSERTION',    'requiresProximity': True,  'maxDistance': 0.40},
    {'classes': frozenset(['person', 'rhino']),    'multiplier': 1.6, 'flag': 'PERSON_NEAR_RHINO',    'requiresProximity': True,  'maxDistance': 0.35},
    {'classes': frozenset(['person', 'elephant']), 'multiplier': 1.4, 'flag': 'PERSON_NEAR_WILDLIFE', 'requiresProximity': True,  'maxDistance': 0.35},
    {'classes': frozenset(['person', 'buffalo']),  'multiplier': 1.3, 'flag': 'PERSON_NEAR_WILDLIFE', 'requiresProximity': True,  'maxDistance': 0.35},
    {'classes': frozenset(['person', 'zebra']),    'multiplier': 1.3, 'flag': 'PERSON_NEAR_WILDLIFE', 'requiresProximity': True,  'maxDistance': 0.35},
]

RISK_CATEGORIES = [
    (0.90, 'CRITICAL'),
    (0.70, 'HIGH THREAT'),
    (0.50, 'ELEVATED'),
    (0.30, 'ACTIVITY DETECTED'),
    (0.00, 'CLEAR'),
]

# Event thresholds
THREAT_OPEN_THRESHOLD  = 0.30   # risk at which a threat event opens
THREAT_CLOSE_THRESHOLD = 0.20   # hysteresis - must drop below this to close
EVENT_COOLDOWN_S       = 8.0    # seconds below close threshold before closing
MAX_EVENT_DURATION_S   = 300.0  # force-close and reopen after 5 minutes

ACTIVITY_INTERVAL_S    = 60.0   # wildlife presence heartbeat interval
HISTORY_WINDOW_S       = 10.0   # temporal risk window
SUSTAINED_THRESHOLD    = 0.60   # avg above this = sustained, scale up
SUSTAINED_MULTIPLIER   = 1.25


def utcNow():
    return datetime.now(timezone.utc).isoformat()


def boxCentre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def normalizedDistance(boxA, boxB, frameShape):
    """
    Center-to-center distance as a fraction of the frame diagonal.
    Returns None if frame dimensions are unknown.
    """
    if not frameShape:
        return None

    height, width = frameShape[0], frameShape[1]
    diagonal = math.hypot(width, height)
    if diagonal == 0:
        return None

    ax, ay = boxCentre(boxA)
    bx, by = boxCentre(boxB)
    return math.hypot(ax - bx, ay - by) / diagonal


######################
## RISK SCORER
######################

class RiskScorer:

    def __init__(self, nodeId='MEERKAT_01', logDir='logs'):
        self.nodeId = nodeId
        self.logDir = logDir
        self.sessionId = datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(logDir, exist_ok=True)
        self.activityLogPath = os.path.join(logDir, f'activity_{self.sessionId}.jsonl')
        self.threatLogPath   = os.path.join(logDir, f'threat_{self.sessionId}.jsonl')

        # (timestamp, score) pairs, trimmed by age not by count
        self.riskHistory = deque()

        self.activeEvent = None
        self.eventCounter = 0
        self.lastActivityLog = 0.0

        print(f"[RISK SCORER] Session: {self.sessionId}")
        print(f"[RISK SCORER] Activity log: {self.activityLogPath}")
        print(f"[RISK SCORER] Threat log:   {self.threatLogPath}")

    ######################
    ## LOGGING
    ######################

    def appendLog(self, path, entry):
        """
        Append-only JSON Lines. One object per line, flushed immediately.
        Crash-safe: a power loss can truncate the last line, never the file.
        Read back with polars.read_ndjson() or json.loads() per line.
        """
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, separators=(',', ':')) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            print(f"[RISK SCORER] Log write failed ({path}): {exc}")

    ######################
    ## SCORING
    ######################

    def getCategory(self, score):
        for threshold, category in RISK_CATEGORIES:
            if score >= threshold:
                return category
        return 'CLEAR'

    def gateDetections(self, detections):
        """
        Split detections into confirmed and advisory. Advisory items are
        below the class gate but close enough to be worth surfacing -
        they never contribute to score but they do get reported.
        """
        confirmed, advisory = [], []

        for det in detections:
            config = CLASS_CONFIG.get(det['class'])
            if config is None:
                continue

            if det['confidence'] >= config['minConf']:
                confirmed.append(det)
            elif det['confidence'] >= config['minConf'] - ADVISORY_BAND:
                advisory.append(det)

        return confirmed, advisory

    def threatWeight(self, det):
        """
        Once a detection clears its gate it carries most of its class
        weight. Confidence above the gate adds a modest bonus rather
        than scaling the whole thing linearly toward zero.
        """
        config = CLASS_CONFIG.get(det['class'])
        if config is None:
            return 0.0

        gate = config['minConf']
        headroom = max(1.0 - gate, 1e-6)
        scale = 0.80 + 0.20 * min((det['confidence'] - gate) / headroom, 1.0)
        return config['baseThreat'] * scale

    def countFactor(self, detections):
        """
        Group size raises risk sublinearly. Capped so a crowded frame
        cannot pin the score on its own.
        """
        threatCount = sum(
            1 for d in detections
            if CLASS_CONFIG.get(d['class'], {}).get('category') == 'threat'
        )

        if threatCount <= 1:
            return 1.0

        return min(1.0 + 0.18 * math.log2(threatCount + 1), 1.5)

    def evaluateCombinations(self, detections, frameShape):
        """
        Returns (flags, multiplier). Proximity-gated combinations only
        fire when the relevant boxes are actually close in frame.
        """
        byClass = {}
        for det in detections:
            byClass.setdefault(det['class'], []).append(det)

        presentClasses = set(byClass)
        flags = []
        multiplier = 1.0

        for combo in COMBINATION_FLAGS:
            if not combo['classes'].issubset(presentClasses):
                continue

            if combo['requiresProximity']:
                classA, classB = tuple(combo['classes'])
                closest = None

                for detA in byClass[classA]:
                    for detB in byClass[classB]:
                        distance = normalizedDistance(
                            detA.get('box'), detB.get('box'), frameShape
                        )
                        if distance is None:
                            continue
                        closest = distance if closest is None else min(closest, distance)

                # Unknown geometry: fire the flag but do not apply the
                # multiplier. Better to surface it than to drop it.
                if closest is None:
                    flags.append(combo['flag'])
                    continue

                if closest > combo['maxDistance']:
                    continue

                # Closer means higher - taper the multiplier by distance.
                proximityScale = 1.0 - (closest / combo['maxDistance']) * 0.35
                effective = 1.0 + (combo['multiplier'] - 1.0) * proximityScale
            else:
                effective = combo['multiplier']

            flags.append(combo['flag'])
            multiplier = max(multiplier, effective)

        return sorted(set(flags)), multiplier

    def score(self, detections, frameShape=None):
        """
        Returns (score, flags, confirmed, advisory).
        """
        confirmed, advisory = self.gateDetections(detections)

        if not confirmed:
            return 0.0, [], [], advisory

        flags, multiplier = self.evaluateCombinations(confirmed, frameShape)
        countScale = self.countFactor(confirmed)

        baseScore = max(self.threatWeight(d) for d in confirmed)
        finalScore = min(baseScore * multiplier * countScale, 1.0)

        return finalScore, flags, confirmed, advisory

    def updateHistory(self, score, now=None):
        """
        Time-based temporal scaling. Sustained elevated risk over the
        last HISTORY_WINDOW_S seconds scales the current score up.
        """
        now = now if now is not None else time.monotonic()
        self.riskHistory.append((now, score))

        cutoff = now - HISTORY_WINDOW_S
        while self.riskHistory and self.riskHistory[0][0] < cutoff:
            self.riskHistory.popleft()

        if not self.riskHistory:
            return score

        average = sum(s for _, s in self.riskHistory) / len(self.riskHistory)

        if average > SUSTAINED_THRESHOLD:
            return min(score * SUSTAINED_MULTIPLIER, 1.0)

        return score

    ######################
    ## EVENT TRACKING
    ######################

    def summarizeDetections(self, detections):
        return [
            {
                'class':      d['class'],
                'confidence': round(d['confidence'], 3),
                'category':   CLASS_CONFIG.get(d['class'], {}).get('category', 'unknown'),
                'box':        [round(v, 1) for v in d.get('box', [])],
            }
            for d in detections
        ]

    def openEvent(self, score, flags, detections, now):
        self.eventCounter += 1
        self.activeEvent = {
            'eventId':        f"{self.nodeId}-{self.sessionId}-{self.eventCounter:04d}",
            'openedAt':       utcNow(),
            'openedMono':     now,
            'peakScore':      score,
            'peakAt':         utcNow(),
            'frames':         1,
            'flags':          set(flags),
            'classes':        {d['class'] for d in detections},
            'maxCount':       len(detections),
            'peakDetections': self.summarizeDetections(detections),
            'belowSince':     None,
        }

    def updateEvent(self, score, flags, detections, now):
        event = self.activeEvent
        event['frames'] += 1
        event['flags'].update(flags)
        event['classes'].update(d['class'] for d in detections)
        event['maxCount'] = max(event['maxCount'], len(detections))

        if score > event['peakScore']:
            event['peakScore'] = score
            event['peakAt'] = utcNow()
            event['peakDetections'] = self.summarizeDetections(detections)

        if score < THREAT_CLOSE_THRESHOLD:
            if event['belowSince'] is None:
                event['belowSince'] = now
        else:
            event['belowSince'] = None

    def closeEvent(self, now, reason='cooldown'):
        event = self.activeEvent
        if event is None:
            return None

        duration = now - event['openedMono']

        record = {
            'schemaVersion': SCHEMA_VERSION,
            'eventId':       event['eventId'],
            'nodeId':        self.nodeId,
            'sessionId':     self.sessionId,
            'type':          'threat',
            'openedAt':      event['openedAt'],
            'closedAt':      utcNow(),
            'durationS':     round(duration, 2),
            'closeReason':   reason,
            'riskAssessment': {
                'peakScore': round(event['peakScore'], 3),
                'category':  self.getCategory(event['peakScore']),
                'peakAt':    event['peakAt'],
            },
            'combinationFlags': sorted(event['flags']),
            'classes':          sorted(event['classes']),
            'maxConcurrent':    event['maxCount'],
            'framesObserved':   event['frames'],
            'peakDetections':   event['peakDetections'],
        }

        self.appendLog(self.threatLogPath, record)
        self.activeEvent = None
        return record

    ######################
    ## MAIN ENTRY POINT
    ######################

    def process(self, detections, frameShape=None, now=None):
        """
        Called from meerkat.py every inference frame.

        Returns a dict of display state. Writes to disk only when an
        event opens, closes, or a wildlife heartbeat interval elapses -
        never once per frame.
        """
        now = now if now is not None else time.monotonic()

        rawScore, flags, confirmed, advisory = self.score(detections, frameShape)
        finalScore = self.updateHistory(rawScore, now)
        category = self.getCategory(finalScore)

        threats = [
            d for d in confirmed
            if CLASS_CONFIG.get(d['class'], {}).get('category') == 'threat'
        ]
        wildlife = [
            d for d in confirmed
            if CLASS_CONFIG.get(d['class'], {}).get('category') == 'wildlife'
        ]

        ######################
        ## THREAT EVENTS
        ######################

        closedEvent = None

        if self.activeEvent is None:
            if finalScore >= THREAT_OPEN_THRESHOLD:
                self.openEvent(finalScore, flags, confirmed, now)
        else:
            self.updateEvent(finalScore, flags, confirmed, now)

            elapsed = now - self.activeEvent['openedMono']
            belowSince = self.activeEvent['belowSince']

            if belowSince is not None and (now - belowSince) >= EVENT_COOLDOWN_S:
                closedEvent = self.closeEvent(now, reason='cooldown')
            elif elapsed >= MAX_EVENT_DURATION_S:
                closedEvent = self.closeEvent(now, reason='maxDuration')
                # Immediately reopen so a long incident stays tracked
                if finalScore >= THREAT_OPEN_THRESHOLD:
                    self.openEvent(finalScore, flags, confirmed, now)

        ######################
        ## WILDLIFE HEARTBEAT
        ######################

        if wildlife and (now - self.lastActivityLog) >= ACTIVITY_INTERVAL_S:
            self.lastActivityLog = now

            speciesCounts = {}
            for d in wildlife:
                speciesCounts[d['class']] = speciesCounts.get(d['class'], 0) + 1

            self.appendLog(self.activityLogPath, {
                'schemaVersion': SCHEMA_VERSION,
                'nodeId':        self.nodeId,
                'sessionId':     self.sessionId,
                'type':          'activity',
                'timestamp':     utcNow(),
                'intervalS':     ACTIVITY_INTERVAL_S,
                'species':       speciesCounts,
                'riskScore':     round(finalScore, 3),
                'detections':    self.summarizeDetections(wildlife),
            })

        return {
            'score':         finalScore,
            'category':      category,
            'flags':         flags,
            'detections':    confirmed,
            'advisory':      advisory,
            'threatCount':   len(threats),
            'wildlifeCount': len(wildlife),
            'eventActive':   self.activeEvent is not None,
            'eventId':       self.activeEvent['eventId'] if self.activeEvent else None,
            'closedEvent':   closedEvent,
        }

    def processAndLog(self, detections, frameShape=None):
        """Convenience wrapper returning just the display triple."""
        state = self.process(detections, frameShape)
        return state['score'], state['category'], state['flags']

    def shutdown(self):
        """Flush any open event so it is not lost on exit."""
        if self.activeEvent is not None:
            self.closeEvent(time.monotonic(), reason='shutdown')