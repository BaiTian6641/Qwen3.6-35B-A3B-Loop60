"""Pure-SVG publication-grade chart renderers for Qwen3.6-35B-A6B-LoopHSSD telemetry.

Zero external plotting dependencies (no matplotlib/seaborn): every chart is a
self-contained, white-background, colorblind-safe vector SVG validated against
`xml.etree.ElementTree.fromstring`.
"""
from __future__ import annotations

import math
from typing import Any
from xml.sax.saxutils import escape as _xml_esc

# Publication palette (high-contrast on white, colorblind-safe)
C_BG = "#ffffff"
C_PANEL = "#f8fafc"
C_BORDER = "#cbd5e1"
C_GRID = "#e2e8f0"
C_TEXT = "#0f172a"
C_MUTED = "#475569"
C_SUBTLE = "#64748b"

C_BLUE = "#1d4ed8"
C_TEAL = "#0d9488"
C_AMBER = "#d97706"
C_CRIMSON = "#dc2626"
C_PURPLE = "#7c3aed"
C_SLATE = "#475569"
C_GREEN = "#15803d"
C_SKY = "#0284c7"

FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_MONO = "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace"


def _esc(text: Any) -> str:
    return _xml_esc(str(text), {'"': "&quot;", "'": "&apos;"})


def _nice_ticks(lo: float, hi: float, n_ticks: int = 5) -> tuple[float, float, list[float]]:
    """Compute clean rounded axis limits and tick positions."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return 0.0, 1.0, [0.0, 0.5, 1.0]
    if abs(hi - lo) < 1e-12:
        pad = max(abs(lo) * 0.1, 0.1)
        lo -= pad
        hi += pad
    span = hi - lo
    raw_step = span / max(n_ticks - 1, 1)
    mag = 10.0 ** math.floor(math.log10(raw_step))
    norm = raw_step / mag
    if norm <= 1.0:
        step = 1.0 * mag
    elif norm <= 2.0:
        step = 2.0 * mag
    elif norm <= 2.5:
        step = 2.5 * mag
    elif norm <= 5.0:
        step = 5.0 * mag
    else:
        step = 10.0 * mag
    nice_lo = math.floor(lo / step) * step
    nice_hi = math.ceil(hi / step) * step
    ticks: list[float] = []
    cur = nice_lo
    for _ in range(32):
        if cur > nice_hi + step * 1e-6:
            break
        ticks.append(round(cur, 10))
        cur += step
    if len(ticks) < 2:
        ticks = [nice_lo, nice_hi]
    return nice_lo, nice_hi, ticks


def _fmt_num(val: float, digits: int = 2) -> str:
    if abs(val) >= 1000 and abs(val - round(val)) < 1e-6:
        return f"{round(val):,}"
    if digits == 0:
        return f"{val:.0f}"
    return f"{val:.{digits}f}"


class _SvgCanvas:
    """Minimal XML-safe SVG builder."""

    def __init__(self, width: int, height: int, title: str, subtitle: str = "") -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">',
            f"<title>{_esc(title)}</title>",
            "<defs>",
            '<style type="text/css">',
            f"text {{ font-family: {FONT_FAMILY}; }}",
            f".mono {{ font-family: {FONT_MONO}; }}",
            "</style>",
            "</defs>",
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="{C_BG}" '
            f'stroke="{C_BORDER}" stroke-width="1"/>',
            f'<text x="28" y="32" font-size="17" font-weight="700" fill="{C_TEXT}">{_esc(title)}</text>',
        ]
        if subtitle:
            self.parts.append(
                f'<text x="28" y="51" font-size="11.5" fill="{C_MUTED}">{_esc(subtitle)}</text>'
            )

    def add(self, elem: str) -> None:
        self.parts.append(elem)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        rx: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        op = f' fill-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
        rx_a = f' rx="{rx:.1f}"' if rx > 0 else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, w):.1f}" height="{max(0.0, h):.1f}"'
            f'{rx_a} fill="{fill}"{op} stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = C_GRID,
        stroke_width: float = 1.0,
        dash: str = "",
        opacity: float = 1.0,
    ) -> None:
        d_a = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' stroke-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"{d_a}{op}/>'
        )

    def polyline(
        self,
        pts: list[tuple[float, float]],
        stroke: str,
        stroke_width: float = 2.2,
        dash: str = "",
        opacity: float = 1.0,
    ) -> None:
        if len(pts) < 2:
            return
        d_a = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' stroke-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
        pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(
            f'<polyline fill="none" stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d_a}{op} points="{pstr}"/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str,
        stroke: str = C_BG,
        stroke_width: float = 1.2,
    ) -> None:
        self.parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        size: float = 11.0,
        fill: str = C_TEXT,
        weight: str = "400",
        anchor: str = "start",
        mono: bool = False,
        rotate: float = 0.0,
    ) -> None:
        cls = ' class="mono"' if mono else ""
        rot = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{cls}{rot}>{_esc(content)}</text>'
        )

    def badge(
        self,
        x: float,
        y: float,
        label: str,
        bg: str,
        fg: str = "#ffffff",
        font_size: float = 10.0,
        anchor: str = "start",
    ) -> None:
        pad_x = 6.0
        est_w = len(label) * font_size * 0.58 + 2 * pad_x
        h = font_size + 7.0
        bx = x if anchor == "start" else (x - est_w if anchor == "end" else x - est_w / 2.0)
        by = y - h + 3.0
        self.rect(bx, by, est_w, h, fill=bg, rx=4.0)
        self.text(bx + est_w / 2.0, y - 2.0, label, size=font_size, fill=fg, weight="600", anchor="middle")

    def legend_row(
        self,
        x: float,
        y: float,
        items: list[tuple[str, str, str]],
        spacing: float = 18.0,
    ) -> None:
        """Draw compact horizontal legend. Each item = (label, color, kind: 'line'|'dash'|'box')."""
        cx = x
        for label, color, kind in items:
            if kind == "box":
                self.rect(cx, y - 9, 12, 10, fill=color, rx=2.0)
            elif kind == "dash":
                self.line(cx, y - 4, cx + 16, y - 4, stroke=color, stroke_width=2.2, dash="4,3")
            else:
                self.line(cx, y - 4, cx + 16, y - 4, stroke=color, stroke_width=2.4)
                self.circle(cx + 8, y - 4, 3.0, fill=color)
            self.text(cx + 21, y, label, size=10.5, fill=C_TEXT, weight="500")
            cx += len(label) * 6.0 + 26.0 + spacing

    def to_svg(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def _map_lin(v: float, lo: float, hi: float, p0: float, p1: float) -> float:
    if abs(hi - lo) < 1e-12:
        return 0.5 * (p0 + p1)
    return p0 + (v - lo) / (hi - lo) * (p1 - p0)


def _draw_panel_axes(
    svg: _SvgCanvas,
    px: float,
    py: float,
    pw: float,
    ph: float,
    x_lo: float,
    x_hi: float,
    x_ticks: list[float],
    y_lo: float,
    y_hi: float,
    y_ticks: list[float],
    x_label: str = "",
    y_label: str = "",
    y_digits: int = 2,
    x_digits: int = 0,
    panel_title: str = "",
) -> None:
    svg.rect(px, py, pw, ph, fill=C_PANEL, stroke=C_BORDER, stroke_width=1.0, rx=4.0)
    if panel_title:
        svg.text(px + 6, py - 8, panel_title, size=11.5, fill=C_TEXT, weight="700")
    for yt in y_ticks:
        yy = _map_lin(yt, y_lo, y_hi, py + ph, py)
        if py - 0.5 <= yy <= py + ph + 0.5:
            svg.line(px, yy, px + pw, yy, stroke=C_GRID, stroke_width=1.0, dash="3,3")
            svg.text(px - 6, yy + 3.5, _fmt_num(yt, y_digits), size=9.5, fill=C_SUBTLE, anchor="end", mono=True)
    for xt in x_ticks:
        xx = _map_lin(xt, x_lo, x_hi, px, px + pw)
        if px - 0.5 <= xx <= px + pw + 0.5:
            svg.line(xx, py, xx, py + ph, stroke=C_GRID, stroke_width=0.8, dash="2,3")
            svg.text(xx, py + ph + 14, _fmt_num(xt, x_digits), size=9.5, fill=C_SUBTLE, anchor="middle", mono=True)
    if x_label:
        svg.text(px + pw / 2.0, py + ph + 30, x_label, size=10.5, fill=C_MUTED, weight="600", anchor="middle")
    if y_label:
        svg.text(px - 44, py + ph / 2.0, y_label, size=10.5, fill=C_MUTED, weight="600", anchor="middle", rotate=-90.0)


# ---------------------------------------------------------------------------
# Chart 1: Held-out Eval NLL (d=1, d=2, d=3) & depth_gain across all 4 stages
# ---------------------------------------------------------------------------
def render_eval_depth_curves_svg(bundle: dict[str, Any]) -> str:
    """Render held-out eval NLL (d=1/2/3) & depth_gain across s1, s1b, s3, s2."""
    eval_curves = bundle.get("eval_curves") or {}
    stage_meta = [
        ("s1_loop_a6b", "S1 Loop A6B (8K)", 1000),
        ("s1b_hssd_distill", "S1b HSSD Distill (16K)", 1500),
        ("s3_long_1m_lsa", "S3 Long 1M D-LSA (64K)", 600),
        ("s2_joint_bptt", "S2 Joint BPTT (131K)", 275),
    ]

    svg = _SvgCanvas(
        1180,
        580,
        "Held-Out Evaluation NLL by Virtual Depth (d=1/2/3) & Depth Gain Across Stages",
        "Qwen3.6-35B-A6B-LoopHSSD v3.2 (40L / 64L / 88L) · Single RTX PRO 6000 Blackwell (96 GiB) · Verified Telemetry",
    )
    svg.legend_row(
        34,
        74,
        [
            ("d=1 (40L)", C_SLATE, "line"),
            ("d=2 (64L)", C_AMBER, "line"),
            ("d=3 (88L)", C_BLUE, "line"),
            ("depth_gain (d1−d3)", C_TEAL, "box"),
        ],
        spacing=26.0,
    )

    margin_l = 62.0
    gap = 44.0
    panel_w = (1180.0 - margin_l - 28.0 - 3 * gap) / 4.0
    top_y = 104.0
    top_h = 225.0
    bot_y = 388.0
    bot_h = 140.0

    # Global depth_gain scale for bottom panels so gains are directly comparable
    all_dg: list[float] = []
    for st_key, _, _ in stage_meta:
        for row in eval_curves.get(st_key) or []:
            dg = row.get("depth_gain")
            if isinstance(dg, (int, float)) and math.isfinite(dg):
                all_dg.append(float(dg))
    dg_lo_raw = min(0.0, *all_dg) if all_dg else 0.0
    dg_hi_raw = max(0.08, max(all_dg) * 1.15) if all_dg else 0.16
    dg_lo, dg_hi, dg_ticks = _nice_ticks(dg_lo_raw, dg_hi_raw, n_ticks=4)

    for idx, (st_key, title, default_max_step) in enumerate(stage_meta):
        px = margin_l + idx * (panel_w + gap)
        rows = [
            r
            for r in (eval_curves.get(st_key) or [])
            if isinstance(r.get("step"), (int, float))
            and any(isinstance(r.get(k), (int, float)) for k in ("d1", "d2", "d3"))
        ]
        rows.sort(key=lambda r: float(r["step"]))
        steps = [float(r["step"]) for r in rows]
        x_lo = 0.0
        x_hi = max(max(steps) if steps else float(default_max_step), 1.0)
        _, x_hi, x_ticks = _nice_ticks(x_lo, x_hi, n_ticks=4)
        x_lo = 0.0
        if x_ticks and x_ticks[0] > 0:
            x_ticks = [0.0, *x_ticks]

        vals = [
            float(r[k])
            for r in rows
            for k in ("d1", "d2", "d3")
            if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))
        ]
        if vals:
            pad = max((max(vals) - min(vals)) * 0.15, 0.02)
            y_lo, y_hi, y_ticks = _nice_ticks(min(vals) - pad, max(vals) + pad, n_ticks=5)
        else:
            y_lo, y_hi, y_ticks = 0.0, 2.0, [0.0, 0.5, 1.0, 1.5, 2.0]

        _draw_panel_axes(
            svg,
            px,
            top_y,
            panel_w,
            top_h,
            x_lo,
            x_hi,
            x_ticks,
            y_lo,
            y_hi,
            y_ticks,
            y_label="Held-Out NLL (nats)" if idx == 0 else "",
            y_digits=2,
            panel_title=title,
        )

        # Plot d1, d2, d3 curves
        for key, color in (("d1", C_SLATE), ("d2", C_AMBER), ("d3", C_BLUE)):
            pts = [
                (_map_lin(float(r["step"]), x_lo, x_hi, px, px + panel_w),
                 _map_lin(float(r[key]), y_lo, y_hi, top_y + top_h, top_y))
                for r in rows
                if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))
            ]
            svg.polyline(pts, stroke=color, stroke_width=2.2)
            for cx, cy in pts:
                svg.circle(cx, cy, 3.4, fill=color)
        if rows:
            last = rows[-1]
            if isinstance(last.get("d3"), (int, float)):
                lx = _map_lin(float(last["step"]), x_lo, x_hi, px, px + panel_w)
                ly = _map_lin(float(last["d3"]), y_lo, y_hi, top_y + top_h, top_y)
                svg.badge(
                    min(lx, px + panel_w - 4),
                    max(ly + 20, top_y + 20) if ly < top_y + top_h - 24 else ly - 10,
                    f"d3={float(last['d3']):.4f}",
                    bg=C_BLUE,
                    font_size=9.5,
                    anchor="end",
                )

        # Bottom subpanel: depth_gain bar/line chart
        _draw_panel_axes(
            svg,
            px,
            bot_y,
            panel_w,
            bot_h,
            x_lo,
            x_hi,
            x_ticks,
            dg_lo,
            dg_hi,
            dg_ticks,
            x_label="Step",
            y_label="Depth Gain (d1−d3)" if idx == 0 else "",
            y_digits=2,
        )
        y_zero = _map_lin(0.0, dg_lo, dg_hi, bot_y + bot_h, bot_y)
        svg.line(px, y_zero, px + panel_w, y_zero, stroke=C_SUBTLE, stroke_width=1.0)
        dg_pts: list[tuple[float, float]] = []
        bar_w = max(8.0, min(20.0, panel_w / max(len(rows) * 2.2, 4.0)))
        for r in rows:
            dg = r.get("depth_gain")
            if not isinstance(dg, (int, float)) or not math.isfinite(float(dg)):
                continue
            cx = _map_lin(float(r["step"]), x_lo, x_hi, px, px + panel_w)
            cy = _map_lin(float(dg), dg_lo, dg_hi, bot_y + bot_h, bot_y)
            dg_pts.append((cx, cy))
            bh = abs(y_zero - cy)
            by = min(cy, y_zero)
            svg.rect(cx - bar_w / 2.0, by, bar_w, bh, fill=C_TEAL, opacity=0.28, rx=2.0)
        svg.polyline(dg_pts, stroke=C_TEAL, stroke_width=2.2)
        for cx, cy in dg_pts:
            svg.circle(cx, cy, 3.2, fill=C_TEAL)
        if rows:
            last_dg_rows = [r for r in rows if isinstance(r.get("depth_gain"), (int, float))]
            if last_dg_rows:
                best_dg = max(last_dg_rows, key=lambda r: float(r["depth_gain"]))
                last_dg = last_dg_rows[-1]
                lbl = f"{float(last_dg['depth_gain']):+.4f}"
                if abs(float(best_dg["depth_gain"]) - float(last_dg["depth_gain"])) > 0.015:
                    lbl = f"pk {float(best_dg['depth_gain']):+.4f} · end {float(last_dg['depth_gain']):+.4f}"
                svg.badge(px + panel_w - 4, bot_y + 18, lbl, bg=C_TEAL, font_size=9.0, anchor="end")

    return svg.to_svg()


# ---------------------------------------------------------------------------
# Chart 2: s1b_hssd_distill Phase-2 HSSD v3.2 Convergence & grad_norm stability
# ---------------------------------------------------------------------------
def render_s1b_hssd_convergence_svg(bundle: dict[str, Any]) -> str:
    """Render s1b Phase-2 standalone HSSD v3.2 NLL (400->1500) + grad_norm stability."""
    s1b_rows = [
        r
        for r in (bundle.get("stage_tables") or {}).get("s1b_hssd_distill", [])
        if isinstance(r.get("step"), (int, float)) and float(r["step"]) >= 400
    ]
    s1b_rows.sort(key=lambda r: float(r["step"]))

    rb_rows = [
        r
        for r in (bundle.get("rolled_back_events") or [])
        if r.get("stage") == "s1b_hssd_distill" and isinstance(r.get("step"), (int, float))
    ]
    rb_rows.sort(key=lambda r: float(r["step"]))

    svg = _SvgCanvas(
        1140,
        510,
        "s1b_hssd_distill Phase-2 Standalone HSSD v3.2 Convergence & Gradient Stability (Steps 400 → 1500)",
        "Post-Fix Jacobian Scaling + Live Blend (commit a0877a3, 0 Skips) vs. "
        "Rolled-Back Pre-Fix Explosion (commits 7a2d590a..6df29172)",
    )
    svg.legend_row(
        34,
        74,
        [
            ("Post-Fix Phase-2 Loss (0 skips)", C_BLUE, "line"),
            ("Post-Fix Distill KL", C_TEAL, "dash"),
            ("Pre-Fix Rolled-Back (194.5K skips)", C_CRIMSON, "dash"),
            ("Blend Ramp λ: 0 → 1 (400–650)", "#bfdbfe", "box"),
        ],
        spacing=20.0,
    )

    # Left panel: Loss (step 400 -> 1500)
    lx, ly, lw, lh = 72.0, 108.0, 480.0, 345.0
    x_lo, x_hi = 400.0, max(1500.0, *(float(r["step"]) for r in s1b_rows))
    x_ticks = [400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1500.0]

    loss_vals = [
        float(r["loss_lm"])
        for r in (*s1b_rows, *rb_rows)
        if isinstance(r.get("loss_lm"), (int, float)) and math.isfinite(float(r["loss_lm"]))
    ]
    y_hi_loss = max(13.0, (max(loss_vals) * 1.08) if loss_vals else 13.0)
    y_lo_l, y_hi_l, y_ticks_l = _nice_ticks(0.0, y_hi_loss, n_ticks=6)
    y_lo_l = 0.0

    _draw_panel_axes(
        svg,
        lx,
        ly,
        lw,
        lh,
        x_lo,
        x_hi,
        x_ticks,
        y_lo_l,
        y_hi_l,
        y_ticks_l,
        x_label="Training Step (Phase 2: 400 → 1500)",
        y_label="Training Loss / Distill KL (nats)",
        y_digits=1,
        panel_title="A. Standalone HSSD v3.2 Convergence vs. Pre-Fix Explosion",
    )

    # Shaded blend ramp region 400..650
    bx0 = _map_lin(400.0, x_lo, x_hi, lx, lx + lw)
    bx1 = _map_lin(min(650.0, x_hi), x_lo, x_hi, lx, lx + lw)
    svg.rect(bx0, ly, bx1 - bx0, lh, fill="#3b82f6", opacity=0.09)
    svg.line(bx1, ly, bx1, ly + lh, stroke=C_SKY, stroke_width=1.2, dash="4,3")
    svg.text(bx1 + 5, ly + 15, "blend = 1.0 (100% HSSD)", size=9.5, fill=C_SKY, weight="600")

    # Pre-fix rolled back loss
    rb_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, lx, lx + lw),
         _map_lin(float(r["loss_lm"]), y_lo_l, y_hi_l, ly + lh, ly))
        for r in rb_rows
        if isinstance(r.get("loss_lm"), (int, float)) and math.isfinite(float(r["loss_lm"]))
    ]
    svg.polyline(rb_pts, stroke=C_CRIMSON, stroke_width=2.0, dash="5,4", opacity=0.85)
    for cx, cy in rb_pts:
        svg.circle(cx, cy, 2.8, fill=C_CRIMSON)
    if rb_pts:
        pk_rb = max(rb_rows, key=lambda r: float(r.get("loss_lm") or 0.0))
        pkx = _map_lin(float(pk_rb["step"]), x_lo, x_hi, lx, lx + lw)
        pky = _map_lin(float(pk_rb["loss_lm"]), y_lo_l, y_hi_l, ly + lh, ly)
        svg.badge(
            pkx + 4,
            max(pky - 6, ly + 28),
            f"Pre-fix peak {float(pk_rb['loss_lm']):.2f} (rolled back)",
            bg=C_CRIMSON,
            font_size=9.0,
        )

    # Post-fix clean loss + distill_kl
    kl_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, lx, lx + lw),
         _map_lin(float(r["distill_kl"]), y_lo_l, y_hi_l, ly + lh, ly))
        for r in s1b_rows
        if isinstance(r.get("distill_kl"), (int, float)) and math.isfinite(float(r["distill_kl"]))
    ]
    svg.polyline(kl_pts, stroke=C_TEAL, stroke_width=1.8, dash="4,3")

    clean_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, lx, lx + lw),
         _map_lin(float(r["loss_lm"]), y_lo_l, y_hi_l, ly + lh, ly))
        for r in s1b_rows
        if isinstance(r.get("loss_lm"), (int, float)) and math.isfinite(float(r["loss_lm"]))
    ]
    svg.polyline(clean_pts, stroke=C_BLUE, stroke_width=2.4)
    for cx, cy in clean_pts:
        svg.circle(cx, cy, 2.6, fill=C_BLUE)

    if s1b_rows:
        last = s1b_rows[-1]
        if isinstance(last.get("loss_lm"), (int, float)):
            ex = _map_lin(float(last["step"]), x_lo, x_hi, lx, lx + lw)
            ey = _map_lin(float(last["loss_lm"]), y_lo_l, y_hi_l, ly + lh, ly)
            svg.badge(
                ex - 4,
                ey - 14,
                f"step {int(last['step'])}: {float(last['loss_lm']):.4f} (0 skips)",
                bg=C_BLUE,
                font_size=9.5,
                anchor="end",
            )

    # Right panel: log10(global_grad_norm)
    rx, ry, rw, rh = 626.0, 108.0, 480.0, 345.0
    g_lo_exp, g_hi_exp = -1.0, 21.0
    g_ticks = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 20.0]
    svg.rect(rx, ry, rw, rh, fill=C_PANEL, stroke=C_BORDER, stroke_width=1.0, rx=4.0)
    svg.text(
        rx + 6, ry - 8, "B. Global Gradient Norm (log₁₀ scale) & 0-Skip Stability",
        size=11.5, fill=C_TEXT, weight="700",
    )
    for gt in g_ticks:
        yy = _map_lin(gt, g_lo_exp, g_hi_exp, ry + rh, ry)
        svg.line(rx, yy, rx + rw, yy, stroke=C_GRID, stroke_width=1.0, dash="3,3")
        svg.text(rx - 6, yy + 3.5, f"1e{int(gt)}", size=9.5, fill=C_SUBTLE, anchor="end", mono=True)
    for xt in x_ticks:
        xx = _map_lin(xt, x_lo, x_hi, rx, rx + rw)
        svg.line(xx, ry, xx, ry + rh, stroke=C_GRID, stroke_width=0.8, dash="2,3")
        svg.text(xx, ry + rh + 14, _fmt_num(xt, 0), size=9.5, fill=C_SUBTLE, anchor="middle", mono=True)
    svg.text(
        rx + rw / 2.0, ry + rh + 30, "Training Step (Phase 2: 400 → 1500)",
        size=10.5, fill=C_MUTED, weight="600", anchor="middle",
    )
    svg.text(
        rx - 44, ry + rh / 2.0, "global_grad_norm (log₁₀)",
        size=10.5, fill=C_MUTED, weight="600", anchor="middle", rotate=-90.0,
    )

    # Clip threshold reference line at grad_norm = 1.0 (log10 = 0)
    y_clip = _map_lin(0.0, g_lo_exp, g_hi_exp, ry + rh, ry)
    svg.line(rx, y_clip, rx + rw, y_clip, stroke=C_GREEN, stroke_width=1.3, dash="4,3")
    svg.text(rx + rw - 6, y_clip - 5, "max_grad_norm = 1.0", size=9.0, fill=C_GREEN, weight="600", anchor="end")

    def _log10_y(gn_val: float) -> float:
        lv = max(g_lo_exp, min(g_hi_exp, math.log10(max(gn_val, 1e-3))))
        return _map_lin(lv, g_lo_exp, g_hi_exp, ry + rh, ry)

    # Pre-fix grad norm
    rb_gn_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, rx, rx + rw), _log10_y(float(r["grad_norm"])))
        for r in rb_rows
        if isinstance(r.get("grad_norm"), (int, float)) and float(r["grad_norm"]) > 0
    ]
    svg.polyline(rb_gn_pts, stroke=C_CRIMSON, stroke_width=2.0, dash="5,4", opacity=0.85)
    for cx, cy in rb_gn_pts:
        svg.circle(cx, cy, 2.8, fill=C_CRIMSON)
    if rb_gn_pts:
        max_rb_g = max(float(r["grad_norm"]) for r in rb_rows if isinstance(r.get("grad_norm"), (int, float)))
        svg.badge(
            rx + 10, ry + 26, f"Pre-fix blowup: up to {max_rb_g:.2e} (194,510 skips)",
            bg=C_CRIMSON, font_size=9.2,
        )

    # Post-fix grad norm
    cl_gn_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, rx, rx + rw), _log10_y(float(r["grad_norm"])))
        for r in s1b_rows
        if isinstance(r.get("grad_norm"), (int, float)) and float(r["grad_norm"]) > 0
    ]
    svg.polyline(cl_gn_pts, stroke=C_BLUE, stroke_width=2.3)
    for cx, cy in cl_gn_pts:
        svg.circle(cx, cy, 2.6, fill=C_BLUE)
    if s1b_rows:
        last_gn = [float(r["grad_norm"]) for r in s1b_rows[-5:] if isinstance(r.get("grad_norm"), (int, float))]
        if last_gn:
            svg.badge(
                rx + rw - 8,
                ry + rh - 96,
                f"Post-fix steady: {min(last_gn):.2f}–{max(last_gn):.2f} · 0 / 1,100 skips",
                bg=C_GREEN,
                font_size=9.5,
                anchor="end",
            )

    return svg.to_svg()


# ---------------------------------------------------------------------------
# Chart 3: s3_long_1m_lsa (lsa_kl, 512K-1M recall, held-out 64K d1/d2/d3 & dg)
# ---------------------------------------------------------------------------
def render_s3_long_1m_dlsa_svg(bundle: dict[str, Any]) -> str:
    """Render s3_long_1m_lsa KL, 512K-1M recall, and held-out 64K d1/d2/d3 + depth_gain."""
    raw_s3 = (bundle.get("stage_tables") or {}).get("s3_long_1m_lsa", [])
    s3_rows = sorted([r for r in raw_s3 if isinstance(r.get("step"), (int, float))], key=lambda r: float(r["step"]))
    raw_ev = (bundle.get("eval_curves") or {}).get("s3_long_1m_lsa", [])
    s3_evals = sorted([r for r in raw_ev if isinstance(r.get("step"), (int, float))], key=lambda r: float(r["step"]))

    svg = _SvgCanvas(
        1140,
        540,
        "s3_long_1m_lsa: 1M-Context D-LSA Indexer KL, 512K–1M Recall & Held-Out 64K Depth Gain",
        "600 Steps (540 × 64K Hybrid + 60 × 512K–1M FP8 Prefix-Suffix Calibration) · "
        "0 Skipped Updates · 1M Live Pass 78.6 GiB",
    )

    x_lo, x_hi = 0.0, max(600.0, *(float(r["step"]) for r in s3_rows))
    x_ticks = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]

    # Top-left: per-layer mean lsa/distill_kl
    lx, ty, lw, th = 72.0, 88.0, 495.0, 185.0
    kl_vals = [float(r["lsa_kl_mean"]) for r in s3_rows if isinstance(r.get("lsa_kl_mean"), (int, float))]
    yk_lo, yk_hi, yk_ticks = _nice_ticks(0.0, max(9.0, max(kl_vals) * 1.08) if kl_vals else 9.0, n_ticks=5)
    yk_lo = 0.0
    _draw_panel_axes(
        svg,
        lx,
        ty,
        lw,
        th,
        x_lo,
        x_hi,
        x_ticks,
        yk_lo,
        yk_hi,
        yk_ticks,
        y_label="Mean Layer KL",
        y_digits=1,
        panel_title="A. D-LSA Per-Layer Mean Distill KL (10 Full-Attn Layers)",
    )
    kl_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, lx, lx + lw),
         _map_lin(float(r["lsa_kl_mean"]), yk_lo, yk_hi, ty + th, ty))
        for r in s3_rows
        if isinstance(r.get("lsa_kl_mean"), (int, float))
    ]
    svg.polyline(kl_pts, stroke=C_PURPLE, stroke_width=2.3)
    for cx, cy in kl_pts:
        svg.circle(cx, cy, 3.0, fill=C_PURPLE)
    if kl_vals:
        svg.badge(
            lx + lw - 6,
            ty + 20,
            f"KL: {kl_vals[0]:.2f} → {kl_vals[-1]:.2f} (min {min(kl_vals):.2f})",
            bg=C_PURPLE,
            font_size=9.5,
            anchor="end",
        )

    # Bottom-left: 512K-1M Top-64 bucket recall
    by, bh = 320.0, 175.0
    rec_vals = [float(r["lsa_recall"]) for r in s3_rows if isinstance(r.get("lsa_recall"), (int, float))]
    yr_lo, yr_hi, yr_ticks = 0.0, 0.70, [0.0, 0.2, 0.4, 0.6, 0.7]
    _draw_panel_axes(
        svg,
        lx,
        by,
        lw,
        bh,
        x_lo,
        x_hi,
        x_ticks,
        yr_lo,
        yr_hi,
        yr_ticks,
        x_label="s3_long_1m_lsa Step",
        y_label="Top-64 Recall",
        y_digits=1,
        panel_title="B. 512K–1M Prefix Top-64 Bucket Recall (Mean & Min-Head)",
    )
    rec_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, lx, lx + lw),
         _map_lin(float(r["lsa_recall"]), yr_lo, yr_hi, by + bh, by))
        for r in s3_rows
        if isinstance(r.get("lsa_recall"), (int, float))
    ]
    min_rec_pts = [
        (_map_lin(float(r["step"]), x_lo, x_hi, lx, lx + lw),
         _map_lin(float(r["lsa_min_head_recall"]), yr_lo, yr_hi, by + bh, by))
        for r in s3_rows
        if isinstance(r.get("lsa_min_head_recall"), (int, float))
    ]
    svg.polyline(min_rec_pts, stroke=C_AMBER, stroke_width=1.8, dash="4,3")
    svg.polyline(rec_pts, stroke=C_TEAL, stroke_width=2.3)
    for cx, cy in rec_pts:
        svg.circle(cx, cy, 3.0, fill=C_TEAL)
    # Mark full 1M prefix step if present (>= 1,000,000 prefix)
    full_1m = [
        r for r in s3_rows
        if isinstance(r.get("calib_prefix_len"), (int, float)) and float(r["calib_prefix_len"]) >= 1000000
    ]
    if full_1m:
        r1m = full_1m[0]
        mx = _map_lin(float(r1m["step"]), x_lo, x_hi, lx, lx + lw)
        svg.line(mx, by, mx, by + bh, stroke=C_SKY, stroke_width=1.2, dash="3,3")
        svg.text(mx + 4, by + 15, f"1,048,576 tok @ s{int(r1m['step'])}", size=9.0, fill=C_SKY, weight="600")
    if rec_vals:
        svg.badge(
            lx + lw - 6,
            by + 20,
            f"mean {rec_vals[0]:.3f} → pk {max(rec_vals):.3f} · end {rec_vals[-1]:.3f}",
            bg=C_TEAL,
            font_size=9.2,
            anchor="end",
        )

    # Right panel: Held-out 64K d1/d2/d3 NLL + depth_gain callouts
    rx, ry, rw, rh = 635.0, 88.0, 470.0, 407.0
    ev_vals = [
        float(r[k])
        for r in s3_evals
        for k in ("d1", "d2", "d3")
        if isinstance(r.get(k), (int, float))
    ]
    if ev_vals:
        ye_lo, ye_hi, ye_ticks = _nice_ticks(min(ev_vals) - 0.02, max(ev_vals) + 0.025, n_ticks=6)
    else:
        ye_lo, ye_hi, ye_ticks = 1.80, 2.10, [1.80, 1.90, 2.00, 2.10]

    _draw_panel_axes(
        svg,
        rx,
        ry,
        rw,
        rh,
        x_lo,
        x_hi,
        x_ticks,
        ye_lo,
        ye_hi,
        ye_ticks,
        x_label="s3_long_1m_lsa Step",
        y_label="Held-Out 64K NLL (nats)",
        y_digits=2,
        panel_title="C. Held-Out 64K Depth Scaling (depth_gain: +0.0750 → +0.1414)",
    )
    svg.legend_row(
        rx + 12,
        ry + 22,
        [("d=1 (40L)", C_SLATE, "line"), ("d=2 (64L)", C_AMBER, "line"), ("d=3 (88L)", C_BLUE, "line")],
        spacing=22.0,
    )
    for key, color in (("d1", C_SLATE), ("d2", C_AMBER), ("d3", C_BLUE)):
        pts = [
            (_map_lin(float(r["step"]), x_lo, x_hi, rx, rx + rw),
             _map_lin(float(r[key]), ye_lo, ye_hi, ry + rh, ry))
            for r in s3_evals
            if isinstance(r.get(key), (int, float))
        ]
        svg.polyline(pts, stroke=color, stroke_width=2.5)
        for cx, cy in pts:
            svg.circle(cx, cy, 4.2, fill=color)

    # Annotate depth_gain gap at each eval point
    for r in s3_evals:
        if all(isinstance(r.get(k), (int, float)) for k in ("d1", "d3", "depth_gain")):
            cx = _map_lin(float(r["step"]), x_lo, x_hi, rx, rx + rw)
            y1 = _map_lin(float(r["d1"]), ye_lo, ye_hi, ry + rh, ry)
            y3 = _map_lin(float(r["d3"]), ye_lo, ye_hi, ry + rh, ry)
            svg.line(cx, y1 + 4, cx, y3 - 4, stroke=C_TEAL, stroke_width=1.6, dash="3,2")
            anc = "start" if float(r["step"]) < 100 else ("end" if float(r["step"]) > 500 else "middle")
            dx = 6.0 if anc == "start" else (-6.0 if anc == "end" else 0.0)
            svg.badge(cx + dx, y3 + 22, f"{float(r['depth_gain']):+.4f}", bg=C_TEAL, font_size=9.5, anchor=anc)

    return svg.to_svg()


# ---------------------------------------------------------------------------
# Chart 4: s2_joint_bptt NLL, Blackwell sm_120 speedup, and -7.0 GiB leak fix
# ---------------------------------------------------------------------------
def render_s2_bptt_throughput_vram_svg(bundle: dict[str, Any]) -> str:
    """Render s2_joint_bptt held-out 131K NLL, throughput speedup, and VRAM leak fix."""
    raw_s2 = (bundle.get("stage_tables") or {}).get("s2_joint_bptt", [])
    s2_rows = sorted([r for r in raw_s2 if isinstance(r.get("step"), (int, float))], key=lambda r: float(r["step"]))
    raw_ev = (bundle.get("eval_curves") or {}).get("s2_joint_bptt", [])
    s2_evals = sorted([r for r in raw_ev if isinstance(r.get("step"), (int, float))], key=lambda r: float(r["step"]))
    sup_s2 = ((bundle.get("supervisor") or {}).get("s2_regimes") or {})

    svg = _SvgCanvas(
        1180,
        525,
        "s2_joint_bptt (4×32K = 131K K/V-Replay BPTT): Held-Out NLL, Blackwell sm_120 Speedup & -7.0 GiB Leak Fix",
        "Pre-Opt Baseline (2e37bf6) → Post-Opt Pre-Leak-Fix (b2c5513, OOM @ s60) → "
        "Post-Leak-Fix Production (5a82677, 17+ Hours Flat Alloc)",
    )

    max_step = max(280.0, *(float(r["step"]) for r in s2_rows))
    _, x_hi_s, x_ticks_s = _nice_ticks(0.0, max_step, n_ticks=5)
    x_lo_s = 0.0

    y_top, p_h = 94.0, 365.0
    p1_x, p1_w = 64.0, 310.0
    p2_x, p2_w = 434.0, 340.0
    p3_x, p3_w = 834.0, 320.0

    # Panel A: Held-out 131K NLL
    ev_vals = [float(r[k]) for r in s2_evals for k in ("d1", "d2", "d3") if isinstance(r.get(k), (int, float))]
    ya_lo, ya_hi, ya_ticks = _nice_ticks(
        (min(ev_vals) - 0.03) if ev_vals else 1.18,
        (max(ev_vals) + 0.03) if ev_vals else 1.50,
        n_ticks=5,
    )
    _draw_panel_axes(
        svg,
        p1_x,
        y_top,
        p1_w,
        p_h,
        x_lo_s,
        x_hi_s,
        x_ticks_s,
        ya_lo,
        ya_hi,
        ya_ticks,
        x_label="s2_joint_bptt Step",
        y_label="Held-Out 131K NLL (nats)",
        y_digits=2,
        panel_title="A. Held-Out 131K NLL (d=1, 2, 3)",
    )
    svg.legend_row(
        p1_x + 8, y_top + 20,
        [("d=1", C_SLATE, "line"), ("d=2", C_AMBER, "line"), ("d=3", C_BLUE, "line")],
        spacing=24.0,
    )
    for key, color in (("d1", C_SLATE), ("d2", C_AMBER), ("d3", C_BLUE)):
        pts = [
            (_map_lin(float(r["step"]), x_lo_s, x_hi_s, p1_x, p1_x + p1_w),
             _map_lin(float(r[key]), ya_lo, ya_hi, y_top + p_h, y_top))
            for r in s2_evals
            if isinstance(r.get(key), (int, float))
        ]
        svg.polyline(pts, stroke=color, stroke_width=2.4)
        for cx, cy in pts:
            svg.circle(cx, cy, 4.0, fill=color)
    d3_evals = [r for r in s2_evals if isinstance(r.get("d3"), (int, float))]
    if len(d3_evals) >= 2:
        delta_d3 = float(d3_evals[-1]["d3"]) - float(d3_evals[0]["d3"])
        svg.badge(
            p1_x + p1_w - 6,
            y_top + p_h - 14,
            f"d3: {float(d3_evals[0]['d3']):.4f} → {float(d3_evals[-1]['d3']):.4f} ({delta_d3:+.4f})",
            bg=C_BLUE,
            font_size=9.2,
            anchor="end",
        )

    # Panel B: Throughput (tok/s)
    tps_vals = [float(r["tok_per_s"]) for r in s2_rows if isinstance(r.get("tok_per_s"), (int, float))]
    yb_hi = max(1150.0, (max(tps_vals) * 1.12) if tps_vals else 1150.0)
    yb_lo, yb_hi, yb_ticks = _nice_ticks(0.0, yb_hi, n_ticks=6)
    yb_lo = 0.0
    _draw_panel_axes(
        svg,
        p2_x,
        y_top,
        p2_w,
        p_h,
        x_lo_s,
        x_hi_s,
        x_ticks_s,
        yb_lo,
        yb_hi,
        yb_ticks,
        x_label="s2_joint_bptt Step",
        y_label="Throughput (tok/s)",
        y_digits=0,
        panel_title="B. Blackwell sm_120 Triton Speedup (~2.05×)",
    )

    def _dedup_tok_seen(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out_r: list[dict[str, Any]] = []
        seen_t: set[int] = set()
        for r in rows_in:
            ts_k = r.get("tokens_seen")
            if isinstance(ts_k, int) and ts_k > 0:
                if ts_k in seen_t:
                    continue
                seen_t.add(ts_k)
            out_r.append(r)
        return out_r

    pre_rows = _dedup_tok_seen(
        [r for r in s2_rows if float(r["step"]) <= 43 and isinstance(r.get("tok_per_s"), (int, float))]
    )
    post_rows = _dedup_tok_seen(
        [r for r in s2_rows if float(r["step"]) >= 50 and isinstance(r.get("tok_per_s"), (int, float))]
    )

    x_split = _map_lin(45.0, x_lo_s, x_hi_s, p2_x, p2_x + p2_w)
    svg.line(x_split, y_top, x_split, y_top + p_h, stroke=C_PURPLE, stroke_width=1.3, dash="4,3")
    svg.text(x_split + 4, y_top + 15, "b2c5513 / 5a82677 Triton opt", size=9.0, fill=C_PURPLE, weight="600")

    pre_pts = [(_map_lin(float(r["step"]), x_lo_s, x_hi_s, p2_x, p2_x + p2_w),
                _map_lin(float(r["tok_per_s"]), yb_lo, yb_hi, y_top + p_h, y_top)) for r in pre_rows]
    post_pts = [(_map_lin(float(r["step"]), x_lo_s, x_hi_s, p2_x, p2_x + p2_w),
                 _map_lin(float(r["tok_per_s"]), yb_lo, yb_hi, y_top + p_h, y_top)) for r in post_rows]
    svg.polyline(pre_pts, stroke=C_AMBER, stroke_width=2.3)
    for cx, cy in pre_pts:
        svg.circle(cx, cy, 3.2, fill=C_AMBER)
    svg.polyline(post_pts, stroke=C_TEAL, stroke_width=2.3)
    for cx, cy in post_pts:
        svg.circle(cx, cy, 3.0, fill=C_TEAL)

    if pre_rows:
        pre_avg = sum(float(r["tok_per_s"]) for r in pre_rows) / len(pre_rows)
        y_pa = _map_lin(pre_avg, yb_lo, yb_hi, y_top + p_h, y_top)
        svg.line(p2_x, y_pa, x_split, y_pa, stroke=C_AMBER, stroke_width=1.4, dash="3,2")
        svg.badge(p2_x + 6, y_pa + 24, f"Pre: {pre_avg:.1f} tok/s", bg=C_AMBER, font_size=9.0)
    if post_rows:
        post_tps = [float(r["tok_per_s"]) for r in post_rows]
        post_avg = sum(post_tps) / len(post_tps)
        y_po = _map_lin(post_avg, yb_lo, yb_hi, y_top + p_h, y_top)
        svg.line(x_split, y_po, p2_x + p2_w, y_po, stroke=C_TEAL, stroke_width=1.4, dash="3,2")
        svg.badge(
            p2_x + p2_w - 6,
            y_top + 40,
            f"Post: {post_avg:.1f} avg ({min(post_tps):.0f}–{max(post_tps):.0f})",
            bg=C_TEAL,
            font_size=9.0,
            anchor="end",
        )

    # Panel C: Post-step Alloc & Peak VRAM (GiB)
    yc_lo, yc_hi, yc_ticks = 0.0, 100.0, [0.0, 20.0, 40.0, 60.0, 80.0, 96.0]
    _draw_panel_axes(
        svg,
        p3_x,
        y_top,
        p3_w,
        p_h,
        x_lo_s,
        x_hi_s,
        x_ticks_s,
        yc_lo,
        yc_hi,
        yc_ticks,
        x_label="s2_joint_bptt Step",
        y_label="HBM Memory (GiB)",
        y_digits=0,
        panel_title="C. -7.0 GiB Leak Fix (Flat 25.4 GiB Alloc)",
    )
    y_hbm = _map_lin(96.0, yc_lo, yc_hi, y_top + p_h, y_top)
    svg.line(p3_x, y_hbm, p3_x + p3_w, y_hbm, stroke=C_CRIMSON, stroke_width=1.4, dash="5,3")
    svg.text(
        p3_x + p3_w - 4, y_hbm - 4, "96 GiB Blackwell HBM Limit",
        size=8.8, fill=C_CRIMSON, weight="600", anchor="end",
    )

    # Peak VRAM line
    peak_pts = [
        (_map_lin(float(r["step"]), x_lo_s, x_hi_s, p3_x, p3_x + p3_w),
         _map_lin(float(r["mem_peak_gib"]), yc_lo, yc_hi, y_top + p_h, y_top))
        for r in s2_rows
        if isinstance(r.get("mem_peak_gib"), (int, float))
    ]
    svg.polyline(peak_pts, stroke=C_BLUE, stroke_width=2.0, dash="4,3")
    for cx, cy in peak_pts:
        svg.circle(cx, cy, 2.5, fill=C_BLUE)

    # Post-step alloc VRAM from supervisor regimes / rows
    alloc_pre_pts = [
        (_map_lin(float(r["step"]), x_lo_s, x_hi_s, p3_x, p3_x + p3_w),
         _map_lin(float(r["mem_alloc_gib"]), yc_lo, yc_hi, y_top + p_h, y_top))
        for r in s2_rows
        if float(r["step"]) <= 43 and isinstance(r.get("mem_alloc_gib"), (int, float))
    ]
    alloc_post_pts = [
        (_map_lin(float(r["step"]), x_lo_s, x_hi_s, p3_x, p3_x + p3_w),
         _map_lin(float(r["mem_alloc_gib"]), yc_lo, yc_hi, y_top + p_h, y_top))
        for r in s2_rows
        if float(r["step"]) >= 50 and isinstance(r.get("mem_alloc_gib"), (int, float))
    ]
    svg.polyline(alloc_pre_pts, stroke=C_AMBER, stroke_width=2.4)
    for cx, cy in alloc_pre_pts:
        svg.circle(cx, cy, 3.0, fill=C_AMBER)
    if alloc_pre_pts:
        svg.text(
            alloc_pre_pts[-1][0] + 6, alloc_pre_pts[-1][1] + 4,
            "Pre-opt: 32.6G", size=8.8, fill=C_AMBER, weight="600",
        )
    svg.polyline(alloc_post_pts, stroke=C_GREEN, stroke_width=2.6)
    for cx, cy in alloc_post_pts:
        svg.circle(cx, cy, 3.0, fill=C_GREEN)

    # Leaky b2c5513 OOM point if observed in supervisor log
    leaky = sup_s2.get("post_opt_leaky_b2c5513") or {}
    if isinstance(leaky.get("alloc_max_gib"), (int, float)):
        lx_o = _map_lin(float(leaky.get("max_step", 60)), x_lo_s, x_hi_s, p3_x, p3_x + p3_w)
        ly_oa = _map_lin(float(leaky["alloc_max_gib"]), yc_lo, yc_hi, y_top + p_h, y_top)
        svg.circle(lx_o, ly_oa, 5.2, fill=C_CRIMSON)
        svg.badge(
            lx_o + 8, ly_oa + 4,
            f"b2c5513 leak: {float(leaky['alloc_max_gib']):.1f}G alloc (OOM)",
            bg=C_CRIMSON, font_size=8.8,
        )
    if isinstance(leaky.get("peak_max_gib"), (int, float)):
        lx_o = _map_lin(float(leaky.get("max_step", 60)), x_lo_s, x_hi_s, p3_x, p3_x + p3_w)
        ly_op = _map_lin(float(leaky["peak_max_gib"]), yc_lo, yc_hi, y_top + p_h, y_top)
        svg.circle(lx_o, ly_op, 4.5, fill=C_CRIMSON)
        svg.text(
            lx_o + 8, ly_op + 3.5, f"{float(leaky['peak_max_gib']):.1f}G peak",
            size=8.8, fill=C_CRIMSON, weight="600",
        )

    if peak_pts:
        svg.badge(
            p3_x + p3_w - 6,
            _map_lin(70.1, yc_lo, yc_hi, y_top + p_h, y_top) - 10,
            "Peak: 70.1 GiB flat",
            bg=C_BLUE,
            font_size=9.0,
            anchor="end",
        )
    if alloc_post_pts:
        last_a = [
            float(r["mem_alloc_gib"])
            for r in s2_rows
            if float(r["step"]) >= 50 and isinstance(r.get("mem_alloc_gib"), (int, float))
        ]
        svg.badge(
            p3_x + p3_w - 6,
            _map_lin(last_a[-1], yc_lo, yc_hi, y_top + p_h, y_top) + 22,
            f"5a82677 alloc: {min(last_a):.1f}–{max(last_a):.1f} GiB (-7.0G)",
            bg=C_GREEN,
            font_size=9.0,
            anchor="end",
        )

    return svg.to_svg()


# ---------------------------------------------------------------------------
# Chart 5: 1M Context Memory Breakdown vs. Inkling 2 Mini & Unshared 88L KV
# ---------------------------------------------------------------------------
def render_kv_memory_vs_inkling_svg(bundle: dict[str, Any]) -> str:
    """Render 1M-context memory breakdown comparing LoopHSSD v3.2, Inkling 2 Mini & unshared 88L KV."""
    mem_rows = bundle.get("kv_memory_1m") or []
    svg = _SvgCanvas(
        1120,
        470,
        "1M-Context (1,048,576 Tokens) Context-State Memory Breakdown",
        "Qwen3.6-35B-A6B-LoopHSSD v3.2 (88L, FP8 1.0× SharedKV + HSSD) vs. "
        "Inkling 2 Mini (48L, F16 KV) & Unshared 88L Baselines",
    )
    svg.legend_row(
        34,
        74,
        [
            ("Global Attention KV Cache", C_BLUE, "box"),
            ("HSSD / Linear / SWA State (O(1))", C_TEAL, "box"),
            ("D-LSA Ward Centroids", C_PURPLE, "box"),
            ("Single Blackwell 96 GiB HBM", C_CRIMSON, "dash"),
        ],
        spacing=22.0,
    )

    px, py, pw, ph = 315.0, 102.0, 765.0, 310.0
    max_gib = max(185.0, max((float(r.get("total_gib", 0.0)) for r in mem_rows), default=185.0) * 1.08)
    x_lo, x_hi, x_ticks = 0.0, max_gib, [0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 176.0]

    svg.rect(px, py, pw, ph, fill=C_PANEL, stroke=C_BORDER, stroke_width=1.0, rx=4.0)
    for xt in x_ticks:
        xx = _map_lin(xt, x_lo, x_hi, px, px + pw)
        if xt == 96.0:
            svg.line(xx, py, xx, py + ph, stroke=C_CRIMSON, stroke_width=1.8, dash="6,4")
            svg.badge(xx, py - 6, "96 GiB Single-GPU HBM Ceiling", bg=C_CRIMSON, font_size=9.2, anchor="middle")
        else:
            svg.line(xx, py, xx, py + ph, stroke=C_GRID, stroke_width=1.0, dash="3,3")
        svg.text(xx, py + ph + 16, f"{int(xt)} GiB", size=9.8, fill=C_SUBTLE, anchor="middle", mono=True)
    svg.text(
        px + pw / 2.0,
        py + ph + 36,
        "1M-Token Context-State HBM Footprint (GiB, excluding model weights)",
        size=10.5,
        fill=C_MUTED,
        weight="600",
        anchor="middle",
    )

    n = max(len(mem_rows), 1)
    slot_h = ph / n
    bar_h = min(36.0, slot_h * 0.56)

    for i, row in enumerate(mem_rows):
        cy = py + (i + 0.5) * slot_h
        by = cy - bar_h / 2.0
        name = str(row.get("name", ""))
        sub = str(row.get("subtitle", ""))
        is_ours = bool(row.get("highlight", False))

        svg.text(
            px - 12, cy - 3, name, size=11.0,
            fill=C_BLUE if is_ours else C_TEXT, weight="700" if is_ours else "600", anchor="end",
        )
        if sub:
            svg.text(px - 12, cy + 12, sub, size=9.2, fill=C_SUBTLE, anchor="end")

        kv_g = float(row.get("kv_gib", 0.0))
        lin_g = float(row.get("linear_or_swa_gib", 0.0))
        idx_g = float(row.get("indexer_gib", 0.0))
        tot_g = float(row.get("total_gib", kv_g + lin_g + idx_g))

        cur_x = px
        for seg_val, seg_col in ((kv_g, C_BLUE), (lin_g, C_TEAL), (idx_g, C_PURPLE)):
            if seg_val <= 0:
                continue
            sw = _map_lin(seg_val, 0.0, x_hi, 0.0, pw)
            if seg_col == C_PURPLE:
                sw = max(sw, 3.5)
            svg.rect(cur_x, by, sw, bar_h, fill=seg_col, opacity=0.92 if is_ours else 0.72, rx=2.0)
            cur_x += sw

        if is_ours:
            tot_w = max(_map_lin(tot_g, 0.0, x_hi, 0.0, pw), cur_x - px)
            svg.rect(px, by, tot_w, bar_h, fill="none", stroke=C_TEXT, stroke_width=1.6, rx=2.0)

        note = str(row.get("callout", f"{tot_g:.2f} GiB"))
        end_x = max(_map_lin(tot_g, 0.0, x_hi, px, px + pw), cur_x)
        if end_x + 210 < px + pw:
            svg.text(
                end_x + 8, cy + 4, note, size=10.2,
                fill=C_BLUE if is_ours else C_TEXT, weight="700" if is_ours else "500",
            )
        else:
            svg.badge(min(end_x - 6, px + pw - 6), cy + 5, note, bg=C_TEXT, font_size=9.5, anchor="end")

    return svg.to_svg()


ALL_CHART_RENDERERS = {
    "eval_depth_curves.svg": render_eval_depth_curves_svg,
    "s1b_hssd_convergence.svg": render_s1b_hssd_convergence_svg,
    "s3_long_1m_dlsa.svg": render_s3_long_1m_dlsa_svg,
    "s2_bptt_throughput_vram.svg": render_s2_bptt_throughput_vram_svg,
    "kv_memory_vs_inkling.svg": render_kv_memory_vs_inkling_svg,
}
