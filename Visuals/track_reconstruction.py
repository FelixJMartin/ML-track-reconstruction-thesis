"""
Particle Track Reconstruction — ML Workflow Animation
======================================================
Requires: pip install manim

Run with:
    manim -pqh track_reconstruction.py TrackReconstruction

Output: media/videos/track_reconstruction/1080p60/TrackReconstruction.mp4
"""

from manim import *
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#0d1117"   # GitHub dark background
HIT_COL     = "#58a6ff"   # detector hits  (blue)
TRACK_COL   = "#3fb950"   # reconstructed tracks (green)
NOISE_COL   = "#f85149"   # noise hits (red)
NET_COL     = "#d2a8ff"   # neural-network nodes (purple)
TEXT_COL    = "#e6edf3"   # labels
DIM_COL     = "#30363d"   # inactive / faded elements
ACCENT      = "#ffa657"   # highlights (orange)


# ── Helpers ───────────────────────────────────────────────────────────────────

def detector_layer(x: float, n_cells: int = 12, height: float = 3.0) -> VGroup:
    """Vertical strip of detector cells."""
    cells = VGroup()
    cell_h = height / n_cells
    for i in range(n_cells):
        y = -height / 2 + (i + 0.5) * cell_h
        rect = Rectangle(width=0.18, height=cell_h * 0.85,
                         fill_color=DIM_COL, fill_opacity=0.6,
                         stroke_color=DIM_COL, stroke_width=0.8)
        rect.move_to([x, y, 0])
        cells.add(rect)
    return cells


def nn_layer(x: float, n: int, col=NET_COL) -> VGroup:
    """Column of neural-network nodes."""
    nodes = VGroup()
    spacing = 0.55
    for i in range(n):
        y = (i - (n - 1) / 2) * spacing
        c = Circle(radius=0.18, fill_color=col,
                   fill_opacity=0.85, stroke_color=WHITE, stroke_width=1)
        c.move_to([x, y, 0])
        nodes.add(c)
    return nodes


def connect_layers(layer_a: VGroup, layer_b: VGroup,
                   col=DIM_COL, opacity=0.25) -> VGroup:
    """Draw edges between every pair of nodes in two layers."""
    edges = VGroup()
    for a in layer_a:
        for b in layer_b:
            line = Line(a.get_center(), b.get_center(),
                        stroke_color=col, stroke_width=0.6,
                        stroke_opacity=opacity)
            edges.add(line)
    return edges


# ── Main scene ────────────────────────────────────────────────────────────────

class TrackReconstruction(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── 0. Title ──────────────────────────────────────────────────────────
        title = Text("Machine Learning for Particle Track Reconstruction",
                     font_size=28, color=TEXT_COL, weight=BOLD)
        subtitle = Text("ATLAS Group · Uppsala University · CERN LHC",
                        font_size=18, color=ACCENT)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        self.play(FadeIn(title_group, shift=UP * 0.3), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(title_group), run_time=0.6)

        # ── 1. LHC detector cross-section ─────────────────────────────────────
        section_label = Text("Step 1 — Raw Detector Data",
                             font_size=22, color=ACCENT).to_edge(UP, buff=0.3)
        self.play(Write(section_label))

        # Five concentric detector layers
        layer_radii = [1.0, 1.5, 2.0, 2.6, 3.2]
        detector_rings = VGroup(*[
            Circle(radius=r, color=DIM_COL, stroke_width=1.5,
                   stroke_opacity=0.5)
            for r in layer_radii
        ])
        detector_rings.shift(LEFT * 3)
        center_dot = Dot(detector_rings.get_center(), color=ACCENT, radius=0.08)
        collision_label = Text("p–p collision", font_size=13,
                               color=ACCENT).next_to(center_dot, DOWN, buff=0.15)

        self.play(Create(detector_rings), FadeIn(center_dot),
                  Write(collision_label), run_time=1.2)
        self.wait(0.4)

        # Simulate ~4 tracks + noise hits
        np.random.seed(42)
        n_tracks = 4
        angles = np.linspace(0.3, 2 * PI - 0.3, n_tracks, endpoint=False)
        origin = detector_rings.get_center()

        hit_dots   = VGroup()
        noise_dots = VGroup()
        track_lines = VGroup()

        for angle in angles:
            for r in layer_radii:
                jitter = np.random.normal(0, 0.06, 2)
                x = origin[0] + r * np.cos(angle) + jitter[0]
                y = origin[1] + r * np.sin(angle) + jitter[1]
                d = Dot([x, y, 0], color=HIT_COL, radius=0.07)
                hit_dots.add(d)

        # Noise hits
        for _ in range(14):
            r     = np.random.choice(layer_radii)
            theta = np.random.uniform(0, 2 * PI)
            jitter = np.random.normal(0, 0.1, 2)
            x = origin[0] + r * np.cos(theta) + jitter[0]
            y = origin[1] + r * np.sin(theta) + jitter[1]
            noise_dots.add(Dot([x, y, 0], color=NOISE_COL,
                               radius=0.055, fill_opacity=0.7))

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in hit_dots],   lag_ratio=0.05),
            LaggedStart(*[GrowFromCenter(d) for d in noise_dots], lag_ratio=0.05),
            run_time=1.5,
        )

        hit_note  = Text("signal hits", font_size=13, color=HIT_COL)
        noise_note = Text("noise hits",  font_size=13, color=NOISE_COL)
        legend = VGroup(
            VGroup(Dot(color=HIT_COL,   radius=0.07), hit_note).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=NOISE_COL, radius=0.07), noise_note).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.next_to(detector_rings, RIGHT, buff=0.35).shift(UP * 0.5)
        self.play(FadeIn(legend))
        self.wait(1.2)

        # ── 2. Neural Network ─────────────────────────────────────────────────
        self.play(FadeOut(section_label))
        section_label2 = Text("Step 2 — Neural Network Processing",
                              font_size=22, color=ACCENT).to_edge(UP, buff=0.3)
        self.play(Write(section_label2))

        # Shrink detector to left
        detector_group = VGroup(detector_rings, center_dot, collision_label,
                                hit_dots, noise_dots, legend)
        self.play(detector_group.animate.scale(0.55).to_edge(LEFT, buff=0.5),
                  run_time=0.8)

        # Build NN
        layer_sizes = [5, 8, 8, 6, 4]
        layer_xs    = np.linspace(0.2, 5.5, len(layer_sizes))
        nn_layers   = [nn_layer(x, n) for x, n in zip(layer_xs, layer_sizes)]
        nn_group    = VGroup(*nn_layers).shift(RIGHT * 0.5)

        edges_group = VGroup()
        for i in range(len(nn_layers) - 1):
            edges_group.add(connect_layers(nn_layers[i], nn_layers[i + 1]))

        nn_labels = VGroup(
            Text("Input\n(hits)", font_size=11, color=TEXT_COL).next_to(nn_layers[0], DOWN, buff=0.2),
            Text("Hidden\nlayers",  font_size=11, color=TEXT_COL).next_to(nn_layers[2], DOWN, buff=0.2),
            Text("Output\n(tracks)", font_size=11, color=TEXT_COL).next_to(nn_layers[4], DOWN, buff=0.2),
        )

        self.play(FadeIn(edges_group), run_time=0.5)
        self.play(
            LaggedStart(*[Create(layer) for layer in nn_layers], lag_ratio=0.15),
            run_time=1.2,
        )
        self.play(FadeIn(nn_labels))

        # Animate "signal" passing through layers
        for layer in nn_layers:
            self.play(
                layer.animate.set_fill(NET_COL, opacity=1.0),
                run_time=0.25,
            )
            self.play(
                layer.animate.set_fill(NET_COL, opacity=0.85),
                run_time=0.15,
            )
        self.wait(0.8)

        # ── 3. Reconstructed tracks ───────────────────────────────────────────
        self.play(FadeOut(section_label2),
                  FadeOut(VGroup(edges_group, *nn_layers, nn_labels)))
        section_label3 = Text("Step 3 — Reconstructed Tracks",
                              font_size=22, color=ACCENT).to_edge(UP, buff=0.3)
        self.play(Write(section_label3))

        # Restore detector, draw clean track lines
        self.play(detector_group.animate.scale(1 / 0.55).move_to(ORIGIN),
                  run_time=0.7)

        for angle in angles:
            end_r = layer_radii[-1] + 0.2
            line = Line(
                origin,
                [origin[0] + end_r * np.cos(angle),
                 origin[1] + end_r * np.sin(angle), 0],
                color=TRACK_COL, stroke_width=2.5,
            )
            track_lines.add(line)

        self.play(
            LaggedStart(*[Create(l) for l in track_lines], lag_ratio=0.15),
            run_time=1.2,
        )

        # Fade noise, highlight signal hits
        self.play(
            noise_dots.animate.set_fill(opacity=0.15).set_stroke(opacity=0.15),
            hit_dots.animate.set_color(TRACK_COL),
            run_time=0.8,
        )
        self.wait(1.0)

        # ── 4. Metrics banner ─────────────────────────────────────────────────
        self.play(FadeOut(section_label3))

        metrics = VGroup(
            Text("Precision", font_size=18, color=TEXT_COL),
            Text("Recall",    font_size=18, color=TEXT_COL),
            Text("Efficiency", font_size=18, color=TEXT_COL),
        ).arrange(RIGHT, buff=1.2)

        values = VGroup(
            Text("94.2 %", font_size=22, color=TRACK_COL, weight=BOLD),
            Text("91.7 %", font_size=22, color=TRACK_COL, weight=BOLD),
            Text("93.0 %", font_size=22, color=TRACK_COL, weight=BOLD),
        ).arrange(RIGHT, buff=1.2)

        board = VGroup(metrics, values).arrange(DOWN, buff=0.2)
        board.to_edge(DOWN, buff=0.6)

        self.play(
            detector_group.animate.scale(0.7).to_edge(UP, buff=0.9),
            track_lines.animate.scale(0.7).move_to(
                detector_group.get_center() + UP * 0.35),
            run_time=0.6,
        )
        self.play(FadeIn(board, shift=UP * 0.2))
        self.wait(2.0)

        # ── 5. Sign-off ───────────────────────────────────────────────────────
        self.play(FadeOut(Group(*self.mobjects)))
        closing = VGroup(
            Text("Felix Martin", font_size=32, color=TEXT_COL, weight=BOLD),
            Text("github.com/FelixJMartin", font_size=18, color=ACCENT),
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(closing, shift=UP * 0.3))
        self.wait(2.5)
        self.play(FadeOut(closing))