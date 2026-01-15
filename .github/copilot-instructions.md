# NP02 Purity Monitor Analysis - AI Agent Guide

This codebase analyzes data from the NP02 liquid argon purity monitor at CERN to extract electron lifetime (tau) and purity measurements.

## Project Architecture

### Data Flow
1. **Raw Data**: Voltage histogram CSVs (F1.txt, F2.txt, F3.txt, F4.txt) in timestamped directories under `/np02data/YYYY_MMM/DD/HH_MM/`
   - F1: Inner long purity monitor (PM)
   - F2: Outer long PM (background for F1)
   - F3: Test pulse signal
   - F4: Short PM signal

2. **Processing Pipeline**:
   - Background subtraction: Scale F2 (background) and subtract from F1 (signal)
   - Peak finding: Locate signal dips/shoulders using derivatives or inverted population
   - Gaussian fitting: Fit custom multi-component Gaussian to extract peak position (m3)
   - Lifetime calculation: `tau = DELTA_T / ln(short_m3 / long_m3)` where DELTA_T varies by electric field

3. **Output**: Plots in `plots*/` directories, cached fit results in `fit_cache*.pkl` and `*.json` files

### Key Analysis Scripts

**Active variants** (understand differences before modifying):
- `np02_analysis_fixed_scaling_october.py`: October 2025+ data with fixed scaling factors, caching strategy
- `TP_analysis.py`, `TP_analysis_CrystallBallFit.py`: Test pulse analysis with voltage region splitting
- `FCdischarges_voltage_scan.py`: Field cage discharge studies with spike detection

**Legacy/experimental**: `np02_analysis_v2.py`, `np02_analysis_dynamicFit.py`, `np02_analysis_dip.py` (keep for reference)

## Critical Patterns

### Custom Gaussian Function
```python
def gaussian(x, m1, m2, m3):
    # Multi-component to model argon scintillation spectrum
    return m1 * (exp(-0.5*((x-m3)^2)/(m2^2)) 
                 + 0.262*exp(-0.5*((x-m3*1.0747)^2)/(m2^2))
                 + 0.077*exp(-0.5*((x-m3*1.0861)^2)/(m2^2)))
```
**Do not** replace with standard Gaussian - coefficients (0.262, 1.0747, etc.) are physics-derived.

### Fixed Scaling Approach
Modern scripts use **fixed** X/Y scaling factors (e.g., `LONG_X_SCALE=1`, `LONG_Y_SCALE=0.57`) instead of optimizing per measurement. This improves stability and speed. When adapting code:
- Respect the `x_scale` and `y_scale` parameters in `process_monitor()`
- Don't reintroduce optimization unless explicitly requested

### Fit Caching Strategy
```python
# Check cache before processing
cache_key = f"{year_month}_{day}_{hour}_{minute}_long"
if cache_key in fit_cache and not FORCE_RERUN:
    return fit_cache[cache_key]
```
Always check `FORCE_RERUN` flag and use `FIT_CACHE_FILE` for expensive fits. Save both pickle (data) and JSON (index).

### Electric Field Context
DELTA_T calculation depends on measurement date:
- **520 V/cm** (default): `DELTA_T = 16.0 / 0.1635`, scaling_factor = 0.92
- **260 V/cm** (Jan 1-10, 2025): `DELTA_T = 16.0 / 0.1104`, scaling_factor = 0.93

Check `get_delta_t_and_scaling_factor()` for boundaries.

### Directory Structure Convention
```python
ROOT_DIR = '/Users/Gajju/NP02_activities/purityMonitor/np02data'
# Expected: np02data/2025_Nov/15/14_30/F1.txt
```
Use `glob.glob(f"{ROOT_DIR}/202*/**/**/**/")` but prefer `iter_measurement_dirs()` (when available) to avoid traversing parent directories.

## Common Tasks

### Adding New Analysis Script
1. Copy from `np02_analysis_fixed_scaling_october.py` (most modern)
2. Update `PLOTS_DIR`, date filters (`START_DATE`, `END_DATE`)
3. Adjust `MIN_BIN_CENTER` if voltage range changes
4. Implement caching with pickle + JSON index
5. Use `process_monitor()` → `process_directory()` → `plot_*()` structure

### Debugging Fit Failures
- Enable `debug = True` to see [INFO]/[SKIP] messages
- Check `fit_cache_errors.json` for systematic issues
- Verify CSV columns: Must have `BinCenter` and `Population`
- Ensure voltage range covers expected peaks (typically 0.4-1.5V)

### Temperature Correlation Analysis
Scripts like `overlay_temperature.py` merge temp data from `Temp.csv` with m3/tau results. Use `pd.merge_asof()` with tolerance for timestamp matching:
```python
merged = pd.merge_asof(sorted_pm, temp_sorted, on='Time', 
                       direction='nearest', tolerance=pd.Timedelta('5min'))
```

## Testing Workflow
1. Set `SAVE_FILES = False` to skip disk I/O during development
2. Process single directory: `process_directory("/path/to/np02data/2025_Nov/15/14_30/")`
3. Enable `debug = True` for verbose output
4. Verify plots in `plots*/` before committing

## Dependencies
Standard scientific stack: `pandas`, `numpy`, `scipy`, `matplotlib`. Physics-specific:
- Crystal Ball function for test pulse analysis (see `TP_analysis_CrystallBallFit.py`)
- LaTeX rendering: Some scripts use `plt.rc('text', usetex=True)` - may need LaTeX installed

## Anti-Patterns
- ❌ Don't use `argrelextrema` on raw data - causes noise sensitivity (legacy approach)
- ❌ Don't re-run fits without checking cache - processing 1000s of files is slow
- ❌ Don't hardcode paths outside `/Users/Gajju/NP02_activities/` - use `ROOT_DIR` variable
- ❌ Don't modify existing cache files during analysis - use `FORCE_RERUN` flag instead
