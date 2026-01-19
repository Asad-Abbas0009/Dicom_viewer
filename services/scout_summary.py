from typing import List, Dict, Any


def build_scout_summary(
    case_id: str,
    scout_plane: str,
    z_start_mm: float,
    z_end_mm: float,
    window_center: float,
    window_width: float,
    kv: int = 120,
    ma: int = 30,
    voice: str = "ON",
    light_timer_sec: int = 5,
    scout_num: int = 1
) -> List[Dict[str, Any]]:
    """
    Build CT-console-style scout summary table.
    """

    return [{
        "scout_num": scout_num,
        "scan_type": "Scout",
        "start_loc_mm": round(z_start_mm, 2),
        "end_loc_mm": round(z_end_mm, 2),
        "kv": kv,
        "ma": ma,
        "scout_plane": scout_plane.upper(),
        "voice": voice,
        "light_timer_sec": light_timer_sec,
        "scout_ww_wl": f"{int(window_width)}/{int(window_center)}"
    }]
