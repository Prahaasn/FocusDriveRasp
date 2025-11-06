# FocusDrive - Speed-Based Activation & Buzzer Enhancement

## Research Summary

### Current State Analysis (✅ Complete)

#### 1. Alert Systems Currently Implemented

**demo_mobilenet.py** (Primary implementation with buzzer):
- ✅ Buzzer alert system ALREADY implemented
- ✅ Plays sound after 3 seconds of sustained distraction at 70% confidence
- ✅ Uses system sounds (macOS: Funk.aiff, Others: system beep)
- ✅ Has cooldown period (5 seconds between alerts)
- ✅ Visual progress bar showing alert status
- ✅ Color-coded alerts (Green=Attentive, Orange=Distracted, Red=Alert)

**demo.py** (LFM model version):
- ✅ Visual alerts (colored overlays, borders)
- ✅ Audio alert option (--alert-sound flag)
- ⚠️ Audio NOT fully implemented (lines 483-486: just a "pass" statement)
- ✅ Confidence threshold system (default 0.7)
- ✅ FPS counter and performance tracking

#### 2. Distraction Duration Tracking

**demo_mobilenet.py** has FULL implementation:
- ✅ `distraction_history` deque (90 frames = 3 seconds at 30 FPS)
- ✅ `sustained_duration` = 3.0 seconds
- ✅ `distraction_threshold` = 0.70 (70% confidence)
- ✅ Tracks last 3 seconds of predictions
- ✅ Triggers when 80% of recent frames show distraction
- ✅ `alert_cooldown` = 5.0 seconds

**demo.py**:
- ❌ NO duration tracking
- Only tracks frame counts (distraction_count, total_frames)
- Shows overall distraction percentage, not sustained detection

#### 3. Speed Detection & GPS/OBD-II Integration

**Current Status**:
- ❌ NO speed detection implemented
- ❌ NO GPS integration
- ❌ NO OBD-II integration
- ❌ NO vehicle speed tracking
- ❌ NO speed-based activation logic

**What's Missing**:
- GPS library integration (gpsd, gps3, or similar)
- OBD-II library integration (python-OBD, obd, or similar)
- Speed threshold monitoring (>15 mph)
- Activation delay tracking (10 seconds above threshold)
- Speed data acquisition module

## Feature Requirements

### Feature 1: Speed-Based Activation
**Goal**: Only activate distraction detection when car is going over 15 mph for more than 10 seconds

**Requirements**:
1. Obtain vehicle speed data (GPS or OBD-II)
2. Monitor speed continuously
3. Track duration above 15 mph threshold
4. Activate detection only after 10+ seconds above threshold
5. Deactivate when speed drops below 15 mph

### Feature 2: Buzzer Alert Enhancement
**Goal**: After 3 seconds of continuous distraction, play a buzzer sound

**Status**: ✅ ALREADY IMPLEMENTED in demo_mobilenet.py
**Note**: demo.py needs implementation (currently has placeholder)

## Implementation Plan

### Phase 1: Understand Current Buzzer Implementation ✅
- [x] Read demo.py alert system
- [x] Read demo_mobilenet.py alert system  
- [x] Check ALERT_SYSTEM.md documentation
- [x] Identify duration tracking mechanism
- [x] Document what exists vs what's needed

### Phase 2: Design Speed Detection Module
- [ ] Research speed data options (GPS vs OBD-II vs both)
- [ ] Choose appropriate Python libraries
- [ ] Design SpeedMonitor class interface
- [ ] Plan integration points with existing demos
- [ ] Define configuration parameters

### Phase 3: Implement Speed Detection
- [ ] Create src/utils/speed_monitor.py module
- [ ] Implement GPS speed detection (if chosen)
- [ ] Implement OBD-II speed detection (if chosen)
- [ ] Add speed threshold monitoring (>15 mph)
- [ ] Add activation delay tracking (10 seconds)
- [ ] Handle fallback/error cases

### Phase 4: Integrate Speed-Based Activation
- [ ] Modify demo_mobilenet.py to include speed monitoring
- [ ] Add --enable-speed-activation flag
- [ ] Add speed display to overlay
- [ ] Only run detection when speed conditions met
- [ ] Add visual indicator for activation status
- [ ] Test with simulated speed data

### Phase 5: Port Buzzer to demo.py (if needed)
- [ ] Copy buzzer implementation from demo_mobilenet.py
- [ ] Adapt for LFM model architecture
- [ ] Test audio alerts
- [ ] Update documentation

### Phase 6: Testing & Documentation
- [ ] Test speed-based activation with real GPS/OBD-II
- [ ] Test buzzer alerts with speed activation
- [ ] Create SPEED_ACTIVATION.md guide
- [ ] Update README.md with new features
- [ ] Add command-line arguments documentation

## Technical Notes

### Speed Data Acquisition Options

**Option 1: GPS (Simpler, Less Accurate)**
- Library: `gpsd` or `gps3`
- Pros: Works anywhere, no car integration needed
- Cons: Less accurate in tunnels/parking garages, slower updates
- Installation: `pip install gpsd-py3` or `pip install gps3`

**Option 2: OBD-II (More Accurate, Requires Hardware)**
- Library: `python-OBD` or `obd`
- Pros: Direct from vehicle, very accurate, real-time
- Cons: Requires OBD-II adapter (ELM327), car-specific
- Installation: `pip install obd`
- Hardware: ~$20 ELM327 Bluetooth/USB adapter

**Option 3: Both (Redundancy)**
- Use OBD-II as primary, GPS as fallback
- Best for production deployment

**Recommendation**: Start with simulated speed for testing, then add real integrations

### Integration Points

**demo_mobilenet.py modifications needed**:
1. Add SpeedMonitor import
2. Initialize speed monitor in `__init__`
3. Add speed check in main loop (line ~416)
4. Skip detection if speed < 15 mph or duration < 10s
5. Add speed overlay to `draw_overlay` method
6. Add activation status indicator

**New configuration parameters**:
```python
self.speed_threshold = 15.0  # mph
self.activation_delay = 10.0  # seconds
self.enable_speed_activation = True  # can be disabled for testing
```

### Existing Code to Leverage

**From demo_mobilenet.py** (lines 106-111):
```python
self.distraction_history = deque(maxlen=90)  # Already tracking time
self.alert_triggered = False
self.last_alert_time = 0
self.alert_cooldown = 5.0
self.distraction_threshold = 0.70
self.sustained_duration = 3.0
```

Can create similar pattern for speed:
```python
self.speed_history = deque(maxlen=300)  # 10 seconds at 30 FPS
self.above_threshold_duration = 0
self.detection_active = False
```

## Dependencies to Add

```txt
# For GPS (Option 1)
gpsd-py3>=0.3.0

# For OBD-II (Option 2)  
obd>=0.7.1
pyserial>=3.5

# Both are optional depending on chosen approach
```

## Review Section
(To be filled after implementation)

---

**Status**: Research complete, ready for implementation planning
**Next Step**: Review this plan with user, then begin Phase 2
**Priority**: Speed detection is new feature, buzzer already works in demo_mobilenet.py
