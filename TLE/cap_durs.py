import json
from datetime import datetime

CAP_START = datetime(2009, 1, 1)
CAP_END = datetime(2026, 9, 1)  # or today

with open('TLE/real_tles.tle.durations.json') as f:
    durations = json.load(f)

capped = {}
for nid, info in durations.items():
    launch_str = info.get('launch')
    decay_str = info.get('decay')
    
    launch = datetime.strptime(launch_str, '%Y-%m-%d') if launch_str else CAP_START
    decay = datetime.strptime(decay_str, '%Y-%m-%d') if decay_str else CAP_END
    
    # Clip to fetch window
    effective_start = max(launch, CAP_START)
    effective_end = min(decay, CAP_END)
    
    dur = (effective_end - effective_start).days
    dur = max(0, dur)
    
    capped[nid] = {
        'launch': launch_str,
        'decay': decay_str,
        'duration_days': dur,
        'original_duration_days': info.get('duration_days', 0),
    }

with open('TLE/real_tles.tle.durations_2009_2026.json', 'w') as f:
    json.dump(capped, f, indent=2)

total_years = sum(v['duration_days'] for v in capped.values()) / 365.25
print(f"Capped total object-years: {total_years:.0f}")
print(f"Original total object-years: {sum(v['duration_days'] for v in durations.values())/365.25:.0f}")
print(f"Ratio: {total_years / (sum(v['duration_days'] for v in durations.values())/365.25):.2%}")