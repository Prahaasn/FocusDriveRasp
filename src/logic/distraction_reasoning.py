"""
Multi-Modal Distraction Reasoning
Combines object detection + distraction classification for robust angle-independent detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DistractionTrigger:
    """Represents a single distraction trigger"""
    name: str
    confidence: float
    description: str
    alert_level: AlertLevel


@dataclass
class DistractionAssessment:
    """Complete distraction assessment result"""
    triggers: List[DistractionTrigger]
    overall_confidence: float
    alert_level: AlertLevel
    is_distracted: bool
    explanation: str


class DistractionReasoning:
    """
    Multi-modal reasoning engine that combines:
    1. Object detection (phone, cup, etc.)
    2. Overall distraction classification (MobileNet)
    3. Spatial reasoning and context
    """

    def __init__(self,
                 object_confidence_threshold: float = 0.6,
                 distraction_confidence_threshold: float = 0.7,
                 enable_object_detection: bool = True,
                 frame_skip: int = 3):
        """
        Initialize reasoning engine

        Args:
            object_confidence_threshold: Minimum confidence for object triggers
            distraction_confidence_threshold: Minimum confidence for distraction classifier
            enable_object_detection: Enable object detection module
            frame_skip: Run object detection every Nth frame (for performance)
        """
        self.object_threshold = object_confidence_threshold
        self.distraction_threshold = distraction_confidence_threshold
        self.enable_object_detection = enable_object_detection
        self.frame_skip = frame_skip

        self.frame_count = 0
        self.last_objects = []  # Cache last object detection results

    def analyze(self,
                frame: np.ndarray,
                distraction_probability: float,
                detected_objects: Optional[List] = None) -> DistractionAssessment:
        """
        Analyze frame and determine distraction status

        Args:
            frame: Input frame (not used directly, but available for future features)
            distraction_probability: Probability from distraction classifier (0-1)
            detected_objects: List of Detection objects from object detector

        Returns:
            DistractionAssessment with triggers and overall decision
        """
        triggers = []

        # Frame counting for skipping
        self.frame_count += 1

        # Use cached objects if skipping this frame
        if detected_objects is not None and (self.frame_count % self.frame_skip == 0 or not self.last_objects):
            self.last_objects = detected_objects
        objects = self.last_objects

        # ===== TRIGGER 1: Overall Distraction Classifier =====
        if distraction_probability > self.distraction_threshold:
            triggers.append(DistractionTrigger(
                name="distracted_posture",
                confidence=distraction_probability,
                description=f"Overall distraction detected ({distraction_probability:.1%})",
                alert_level=AlertLevel.MEDIUM
            ))

        # ===== TRIGGER 2: Phone Detection =====
        if self.enable_object_detection and objects:
            phone_detections = [obj for obj in objects if obj.class_name == 'cell phone']
            if phone_detections:
                max_conf = max(obj.confidence for obj in phone_detections)
                if max_conf > self.object_threshold:
                    triggers.append(DistractionTrigger(
                        name="phone_use",
                        confidence=max_conf,
                        description=f"Phone detected in driver area ({max_conf:.1%})",
                        alert_level=AlertLevel.HIGH
                    ))

            # ===== TRIGGER 3: Eating/Drinking =====
            eating_objects = [obj for obj in objects
                            if obj.class_name in ['cup', 'bottle', 'wine glass']]
            if eating_objects:
                max_conf = max(obj.confidence for obj in eating_objects)
                if max_conf > self.object_threshold:
                    triggers.append(DistractionTrigger(
                        name="eating_drinking",
                        confidence=max_conf,
                        description=f"Eating/drinking detected ({max_conf:.1%})",
                        alert_level=AlertLevel.MEDIUM
                    ))

            # ===== TRIGGER 4: Other Distracting Objects =====
            other_objects = [obj for obj in objects
                           if obj.class_name in ['book', 'laptop']]
            if other_objects:
                max_conf = max(obj.confidence for obj in other_objects)
                if max_conf > self.object_threshold:
                    triggers.append(DistractionTrigger(
                        name="distracting_object",
                        confidence=max_conf,
                        description=f"Distracting object detected: {other_objects[0].class_name}",
                        alert_level=AlertLevel.MEDIUM
                    ))

        # ===== MULTI-MODAL FUSION =====
        assessment = self._fuse_triggers(triggers, distraction_probability)

        return assessment

    def _fuse_triggers(self, triggers: List[DistractionTrigger],
                      distraction_prob: float) -> DistractionAssessment:
        """
        Fuse multiple triggers into final distraction assessment

        Uses weighted voting with priority given to high-confidence signals
        """
        if not triggers:
            # No triggers - driver is attentive
            return DistractionAssessment(
                triggers=[],
                overall_confidence=1.0 - distraction_prob,
                alert_level=AlertLevel.NONE,
                is_distracted=False,
                explanation="Driver is attentive"
            )

        # Calculate weighted confidence
        # Strategy: Use maximum confidence from highest priority trigger
        # If multiple high-level triggers, boost confidence

        high_triggers = [t for t in triggers if t.alert_level == AlertLevel.HIGH]
        medium_triggers = [t for t in triggers if t.alert_level == AlertLevel.MEDIUM]

        if high_triggers:
            # Phone use or other critical distraction
            max_conf = max(t.confidence for t in high_triggers)
            # Boost if multiple triggers agree
            if len(triggers) > 1:
                max_conf = min(0.99, max_conf * 1.1)

            alert_level = AlertLevel.HIGH
            overall_conf = max_conf

        elif medium_triggers:
            # Eating, drinking, or posture-based distraction
            max_conf = max(t.confidence for t in medium_triggers)
            # Boost if multiple medium triggers
            if len(medium_triggers) > 1:
                max_conf = min(0.95, max_conf * 1.05)

            alert_level = AlertLevel.MEDIUM
            overall_conf = max_conf

        else:
            # Low-level triggers only
            max_conf = max(t.confidence for t in triggers)
            alert_level = AlertLevel.LOW
            overall_conf = max_conf

        # Generate explanation
        explanation = self._generate_explanation(triggers)

        return DistractionAssessment(
            triggers=triggers,
            overall_confidence=overall_conf,
            alert_level=alert_level,
            is_distracted=True,
            explanation=explanation
        )

    def _generate_explanation(self, triggers: List[DistractionTrigger]) -> str:
        """Generate human-readable explanation of distraction"""
        if not triggers:
            return "Driver is attentive"

        # Sort by alert level (highest first)
        sorted_triggers = sorted(triggers, key=lambda t: t.alert_level.value, reverse=True)

        # Primary reason
        primary = sorted_triggers[0]
        explanation = primary.description

        # Add secondary reasons if multiple triggers
        if len(triggers) > 1:
            secondary = sorted_triggers[1]
            explanation += f" + {secondary.name.replace('_', ' ')}"

        if len(triggers) > 2:
            explanation += f" + {len(triggers)-2} more"

        return explanation

    def get_trigger_summary(self, assessment: DistractionAssessment) -> Dict:
        """Get summary of active triggers for display"""
        return {
            'total_triggers': len(assessment.triggers),
            'high_priority': len([t for t in assessment.triggers if t.alert_level == AlertLevel.HIGH]),
            'medium_priority': len([t for t in assessment.triggers if t.alert_level == AlertLevel.MEDIUM]),
            'trigger_names': [t.name for t in assessment.triggers],
            'explanation': assessment.explanation
        }


class DistrationReasoningLite:
    """
    Lightweight version of distraction reasoning
    Uses simple rule-based logic without complex fusion
    """

    def __init__(self, phone_threshold: float = 0.7, distraction_threshold: float = 0.7):
        self.phone_threshold = phone_threshold
        self.distraction_threshold = distraction_threshold

    def analyze_simple(self, distraction_prob: float, has_phone: bool) -> Tuple[bool, str]:
        """
        Simple binary decision: distracted or not

        Args:
            distraction_prob: Probability from classifier
            has_phone: Whether phone was detected

        Returns:
            (is_distracted, reason)
        """
        if has_phone:
            return True, "Phone detected"
        elif distraction_prob > self.distraction_threshold:
            return True, f"Distracted posture ({distraction_prob:.1%})"
        else:
            return False, "Attentive"


if __name__ == "__main__":
    # Test the reasoning engine
    from models.object_detector import Detection

    # Create reasoning engine
    reasoning = DistractionReasoning()

    # Simulate detection results
    print("=== Test Case 1: Phone detected + High distraction probability ===")
    phone = Detection(77, 'cell phone', 0.85, (100, 100, 50, 50))
    assessment = reasoning.analyze(
        frame=None,
        distraction_probability=0.92,
        detected_objects=[phone]
    )
    print(f"Is distracted: {assessment.is_distracted}")
    print(f"Confidence: {assessment.overall_confidence:.2%}")
    print(f"Alert level: {assessment.alert_level.name}")
    print(f"Explanation: {assessment.explanation}")
    print(f"Triggers: {[t.name for t in assessment.triggers]}")

    print("\n=== Test Case 2: No objects, medium distraction ===")
    assessment = reasoning.analyze(
        frame=None,
        distraction_probability=0.75,
        detected_objects=[]
    )
    print(f"Is distracted: {assessment.is_distracted}")
    print(f"Confidence: {assessment.overall_confidence:.2%}")
    print(f"Alert level: {assessment.alert_level.name}")
    print(f"Explanation: {assessment.explanation}")

    print("\n=== Test Case 3: Cup detected, low distraction ===")
    cup = Detection(47, 'cup', 0.82, (200, 150, 40, 60))
    assessment = reasoning.analyze(
        frame=None,
        distraction_probability=0.45,
        detected_objects=[cup]
    )
    print(f"Is distracted: {assessment.is_distracted}")
    print(f"Confidence: {assessment.overall_confidence:.2%}")
    print(f"Alert level: {assessment.alert_level.name}")
    print(f"Explanation: {assessment.explanation}")
    print(f"Triggers: {[t.name for t in assessment.triggers]}")
