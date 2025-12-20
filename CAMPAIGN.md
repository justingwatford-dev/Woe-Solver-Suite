# Long Haul Campaign Status

Hurricane lifecycle validation campaign for Oracle V4.

**Campaign Version:** Long Haul (December 2025)  
**Goal:** Validate against 15 diverse Atlantic hurricanes

---


💻 Usage Examples

Mode C (Reconstruction) - Default
bashpython run_long_haul_template.py \
    --storm KATRINA --year 2005 \
    --frames 150000 \
    --mode reconstruction

Mode B (Assisted)
bashpython run_long_haul_template.py \
    --storm RITA --year 2005 \
    --frames 170000 \
    --mode assisted

Mode A (Free Run)
bashpython run_long_haul_template.py \
    --storm WILMA --year 2005 \
    --frames 250000 \
    --mode free_run

bash# Control
python run_long_haul_template.py \
    --storm WILMA --year 2005 --frames 250000

# Ablation
python run_long_haul_template.py \
    --storm WILMA --year 2005 --frames 250000 --ablation

---

## Storm Roster

| # | Storm | Year | Category | Frames | Status |
|---|-------|------|----------|--------|--------|
| 1 | Charley | 2004 | 4 | 120,000 | ✅ Complete |
| 2 | Sandy | 2012 | 3 | 175,000 | ✅ Complete |
| 3 | Ivan | 2004 | 5 | 300,000 | 🔄 Active |
| 4 | Dennis | 2005 | 4 | TBD | ⏳ Queued |
| 5 | Ike | 2008 | 4 | TBD | ⏳ Queued |
| 6 | Isabel | 2003 | 5 | TBD | ⏳ Queued |
| 7 | Wilma | 2005 | 5 | TBD | ⏳ Queued |
| 8 | Jeanne | 2004 | 3 | TBD | ⏳ Queued |
| 9 | Ophelia | 2005 | 1 | TBD | ⏳ Queued |
| 10 | Irene | 2011 | 3 | TBD | ⏳ Queued |
| 11 | Rita | 2005 | 5 | TBD | ⏳ Queued |
| 12 | Emily | 2005 | 5 | TBD | ⏳ Queued |
| 13 | Opal | 1995 | 4 | TBD | ⏳ Queued |
| 14 | Felix | 2007 | 5 | TBD | ⏳ Queued |
| 15 | Dean | 2007 | 5 | TBD | ⏳ Queued |

---

## Completed Storms

### Hurricane Charley (2004) ✅
**Date:** December 2, 2025  
**Duration:** 120,000 frames (5.5 days simulated, 7.12 hours wall time)

| Metric | Result |
|--------|--------|
| Track RMSE | 110.59 km (Haversine) |
| Landfall Error | <15 km (direct hit) |
| Landfall Location | 26.9°N, 82.1°W (Punta Gorda) |
| Peak Intensity | 111 kts (Cat 3/4) |
| Navigation Confidence | 77.9% |
| Interventions | 301 (8 helpful) |

**Notes:** Genesis Governor successfully prevented early hypercane. Single spike to 200.7 kts at frame 65,800 (Phoenix Protocol recovery confirmed). Pre-V30 baseline run.

---

### Hurricane Sandy (2012) ✅
**Date:** December 2-3, 2025  
**Duration:** 175,000 frames (8+ days simulated, ~12-15 hours wall time)

**Challenges:**
- Massive wind field (900+ mile tropical storm winds)
- Extratropical transition
- Famous "left turn" into New Jersey
- Longest simulation ever attempted at that time

---

### Hurricane Ivan (2004) 🔄 IN PROGRESS
**Launch:** December 13, 2025  
**Duration:** 300,000 frames (increased from 220k)

**Current Status:**
- Latest position: 19.5°N, -82.8°W (western Cuba)
- Tracking beyond Cayman Islands! 🎉
- Testing V52 Lazarus Protocol

**Challenges:**
- Erratic loop-de-loop historical track
- Cayman stall zone (28-hour parking in reality)
- Extended simulation for Alabama landfall

---

## Storm Tiers

### Tier 1: Intensity Diversity
| Storm | Feature |
|-------|---------|
| Dennis (2005) | Rapid intensification specialist |
| Ike (2008) | Annular structure, persistent |
| Isabel (2003) | Long-lived Cape Verde system |

### Tier 2: Track Complexity
| Storm | Feature |
|-------|---------|
| Ivan (2004) | Erratic loop-de-loop |
| Wilma (2005) | Sharp recurvature |
| Jeanne (2004) | Caribbean meandering |
| Ophelia (2005) | Coastal paralleling |
| Irene (2011) | Long coastal track |

### Tier 3: Structural Phenomena
| Storm | Feature |
|-------|---------|
| Rita (2005) | Eyewall replacement cycle |
| Emily (2005) | Multiple ERCs |
| Opal (1995) | Rapid intensification then collapse |
| Felix (2007) | Extreme rapid intensification |
| Dean (2007) | Long-lived Cat 5 |

---

## Validation Protocol

### For Each Storm:

**Phase 1: Pre-Run**
- Historical summary
- ERA5 data verification
- Initial conditions
- Expected challenges

**Phase 2: Execution**
- Full lifecycle (genesis → landfall)
- Real-time metrics
- Oracle intervention logging
- Phoenix Protocol monitoring

**Phase 3: Analysis**
- Track overlay (HURDAT2 vs Oracle)
- Intensity timeline comparison
- Intervention effectiveness
- Failure mode identification

**Phase 4: Lessons**
- Physics documentation
- Oracle patterns
- Edge cases
- Patch requirements

---

## Success Criteria

### Quantitative
- Mean track RMSE < 50 km at 72h forecast
- Intensity error < ±15 kts (mean absolute)
- Structural features captured > 80%
- Zero hypercane formations

### Qualitative
- Code maintainable by external developers
- Documentation enables reproduction
- Edge cases clearly identified
- UI usable by non-programmers (future)

---

**Target Completion:** February 2026

*For patch details, see [PATCHES.md](./PATCHES.md)*
