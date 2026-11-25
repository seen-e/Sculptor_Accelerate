"""Shared suction pick / place detection + HTML writer for make_suction_report.py and make_bag_report.py.

Signals (all sampled on the same uniform grid t):
  p      vacuum pressure; the vacuum is always on, idle (nothing attached) sits at a baseline,
         a box attached drives it well below that.
  suc_a  suction command (1 = vacuum on, 0 = release pulse = place)
  suc_s  suction DO status (follows suc_a with a small lag)
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_PLOTLY = "/home/seene/miniconda3/envs/genesis/lib/python3.12/site-packages/plotly/package_data/plotly.min.js"
CDN = '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>'


def r3(a):
    return np.round(np.asarray(a, dtype=float), 3).tolist()


def edges(x):
    d = np.diff(np.asarray(x).astype(int))
    return np.where(d > 0)[0] + 1, np.where(d < 0)[0] + 1


def spans(mask):
    m = np.concatenate([[0], np.asarray(mask).astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def merged_spans(mask, gap):
    """spans of mask with gaps of <= gap samples closed (pressure hovering around a threshold)."""
    out = []
    for a0, b0 in spans(mask):
        if out and a0 - out[-1][1] - 1 <= gap:
            out[-1][1] = b0
        else:
            out.append([a0, b0])
    return out


def idle_baseline(p, suc_a):
    """Idle vacuum level: the most common pressure while the command is on (the box-attached plateaus are
    shorter than the idle stretches and spread over many levels)."""
    v = np.round(np.asarray(p)[np.asarray(suc_a) > 0])
    v = v[v < -3]
    vals, cnt = np.unique(v, return_counts=True)
    return float(vals[np.argmax(cnt)])


def detect_cycles(t, p, suc_a, suc_s, fps, pick, weak, released=-5.0, src_t=None):
    n = len(t)
    st = (lambda i: float(src_t[i])) if src_t is not None else (lambda i: None)
    gap = int(round(0.3 * fps))
    merged = merged_spans(p < pick, gap)
    weak_merged = merged_spans(p < weak, gap)
    rise_a, fall_a = edges(suc_a)   # command 1->0 = release (place)
    rise_s, fall_s = edges(suc_s)
    cycles, used = [], set()
    for fa in fall_a:
        ra = rise_a[rise_a > fa]
        end = int(ra[0]) if len(ra) else n - 1
        prev_end = int(rise_a[rise_a <= fa][-1]) if (rise_a <= fa).any() else -1
        # the hold span that is still active right before the release command
        pk, is_weak = None, False
        for si, (a0, b0) in enumerate(merged):
            if a0 <= fa - 1 <= b0 + 3 and a0 > prev_end:
                pk = int(a0); used.add(si); break
        if pk is None:
            for a0, b0 in weak_merged:
                if a0 <= fa - 1 <= b0 + 3 and a0 > prev_end:
                    pk, is_weak = int(a0), True; break
        rel = np.where(p[fa:] > released)[0]
        rel_i = int(fa + rel[0]) if len(rel) else None
        fs = fall_s[fall_s >= fa - 5]
        lo = pk if pk is not None else max(prev_end, 0)
        cycles.append(dict(
            pick=pk, weak=is_weak, place=int(fa), place_end=end,
            place_state=int(fs[0]) if len(fs) else None,
            vac_released=rel_i,
            pick_t=None if pk is None else round(float(t[pk]), 3),
            place_t=round(float(t[fa]), 3), place_end_t=round(float(t[end]), 3),
            hold_s=None if pk is None else round(float(t[fa] - t[pk]), 3),
            release_s=round(float(t[end] - t[fa]), 3),
            vac_release_s=None if rel_i is None else round(float(t[rel_i] - t[fa]), 3),
            peak_vac=round(float(p[lo:fa + 1].min()), 1),
            pick_src=None if pk is None else st(pk), place_src=st(int(fa)), place_end_src=st(end),
        ))
    # holds longer than 0.5 s that never ended in a place command
    orphan = [int(a0) for si, (a0, b0) in enumerate(merged) if si not in used and (b0 - a0 + 1) > 0.5 * fps]
    return dict(
        cycles=cycles, orphan_picks=orphan,
        hold_spans=[(c["pick"], c["place"]) for c in cycles if c["pick"] is not None],
        place_spans=[(c["place"], c["place_end"]) for c in cycles],
    )


def write_html(meta, episodes, out, plotly_js=DEFAULT_PLOTLY):
    html = (Path(__file__).with_name("suction_report_template.html").read_text()
            .replace("/*__DATA__*/null", json.dumps(dict(meta=meta, episodes=episodes), ensure_ascii=False,
                                                    separators=(",", ":"))))
    pj = Path(plotly_js) if plotly_js else None
    if pj and pj.exists():
        html = html.replace(CDN, "<script>" + pj.read_text().replace("</script", "<\\/script") + "</script>")
    else:
        print("plotly.min.js not found, using CDN (needs internet)")
    out = Path(out)
    out.write_text(html)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
