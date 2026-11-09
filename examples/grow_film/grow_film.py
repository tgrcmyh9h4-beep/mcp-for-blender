"""What If You Grew and Couldn't Stop?  --  a 60 s low-poly short, built entirely in bpy.

The camera is locked looking straight down (-Z) and only ever moves up the Z axis,
from 1.7 m above a street to 5,000 km above a toy solar system.

Run headless (EEVEE needs a GL context, so wrap in xvfb on a machine with no display):

    xvfb-run -a blender -b --python grow_film.py -- --stills 1,500,1440 --pct 50
    xvfb-run -a blender -b --python grow_film.py -- --animation            # PNG sequence
    blender --python grow_film.py                                          # just build, in the UI

Options (after the "--"):
    --out DIR          output directory (default: ./renders next to this file)
    --stills A,B,C     render these frames as PNG stills
    --animation        render frames --start..--end as a PNG sequence (resumable)
    --start N --end N  frame range for --animation (default 1..1440)
    --pct N            resolution percentage of 1920x1080 (default 100)
    --samples N        EEVEE TAA samples (default 16)
    --engine NAME      BLENDER_EEVEE (default) or CYCLES
    --no-volume        skip the volumetric cloud slab (mesh clouds only)
    --save-blend PATH  save the generated scene

Scale note: distances are compressed so the whole trip fits in one scene.  The street,
tower and clouds are true scale; the "Earth" is a 40 km ball, the Moon orbits ~100 km
out and the Sun sits 1,200 km away.  The camera's altitude grows exponentially, so
this reads as a continuous pull-back.
"""
import argparse
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Vector, noise

# ----------------------------------------------------------------------------- config
FPS = 24
DURATION_S = 60
F_START, F_END = 1, FPS * DURATION_S  # 1..1440

# (time s, camera altitude m) -- interpolated smoothly in log space
ALTITUDE_KNOTS = [
    (0, 1.7),       # eye height on the street
    (10, 45),       # above the roofs
    (25, 420),      # above the top of the tower
    (40, 15_000),   # above the clouds, horizon starts to curve
    (45, 60_000),   # Earth is a ball
    (50, 250_000),  # Earth shrinking, Moon in view
    (60, 5_000_000),  # solar system
]

EARTH_R = 40_000.0
EARTH_SINK = 30.0            # Earth's pole sits this far under the city plane
CITY_HALF = 600.0
TOWER = dict(x0=9.0, x1=41.0, y0=-10.0, y1=26.0, h=320.0)
CLOUD_BASE, CLOUD_TOP = 1500.0, 2800.0
SUN_POS = Vector((-1_200_000.0, 0.0, -EARTH_R))
SUN_R = 260_000.0
# direction *towards* the sun: high and mostly along the street, so the street stays sunlit
SUN_DIR = Vector((-0.28, -0.45, 0.85)).normalized()

random.seed(7)


def frame_of(t):
    return int(round(1 + t * FPS))


# ----------------------------------------------------------------------------- maths
def pchip(xs, ys):
    """Monotone cubic interpolation (Fritsch-Carlson) so altitude never dips."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def f(x):
        x = min(max(x, xs[0]), xs[-1])
        i = 0
        while i < n - 2 and x > xs[i + 1]:
            i += 1
        t = (x - xs[i]) / h[i]
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
                + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1])
    return f


_alt = pchip([k[0] for k in ALTITUDE_KNOTS], [math.log(k[1]) for k in ALTITUDE_KNOTS])


def altitude(frame):
    return math.exp(_alt((frame - 1) / FPS))


def on_earth(x, y, h=0.0):
    """Z of a point at height h above the (true) Earth sphere, at lateral (x, y)."""
    return math.sqrt(max((EARTH_R + h) ** 2 - x * x - y * y, 0.0)) - EARTH_R - EARTH_SINK


# ----------------------------------------------------------------------------- materials
def _set(bsdf, names, value):
    for n in names:
        if n in bsdf.inputs:
            bsdf.inputs[n].default_value = value
            return


def make_mat(name, color, emit=0.0, emit_color=None, rough=1.0, spec=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    _set(b, ("Specular IOR Level", "Specular"), spec)
    if emit:
        _set(b, ("Emission Color", "Emission"), (*(emit_color or color), 1.0))
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.blend_method = 'BLEND'
        m.shadow_method = 'NONE'
    m.diffuse_color = (*color, 1.0)
    return m


MATS = {}


def M(name):
    return MATS[name]


def build_materials():
    spec = {
        "grass": (0.28, 0.52, 0.20), "forest": (0.13, 0.36, 0.16), "field_a": (0.45, 0.62, 0.22),
        "field_b": (0.66, 0.62, 0.30), "field_c": (0.36, 0.47, 0.18), "desert": (0.76, 0.66, 0.42),
        "ocean": (0.07, 0.26, 0.62), "lake": (0.12, 0.38, 0.70), "ice": (0.93, 0.96, 1.0),
        "asphalt": (0.17, 0.17, 0.19), "asphalt_dark": (0.12, 0.12, 0.14),
        "asphalt_light": (0.22, 0.22, 0.24), "manhole": (0.20, 0.20, 0.22), "sidewalk": (0.62, 0.61, 0.58), "paint": (0.95, 0.95, 0.9),
        "wall_a": (0.93, 0.87, 0.74), "wall_b": (0.80, 0.64, 0.52), "wall_c": (0.72, 0.80, 0.86),
        "wall_d": (0.90, 0.90, 0.92),
        "roof_a": (0.70, 0.22, 0.16), "roof_b": (0.33, 0.24, 0.20), "roof_c": (0.27, 0.33, 0.42),
        "trunk": (0.36, 0.23, 0.13), "leaf_a": (0.20, 0.50, 0.18), "leaf_b": (0.30, 0.60, 0.20),
        "leaf_c": (0.12, 0.38, 0.20),
        "metal": (0.30, 0.31, 0.34), "tower": (0.55, 0.60, 0.66), "tower_trim": (0.32, 0.36, 0.42),
        "bld_a": (0.62, 0.60, 0.58), "bld_b": (0.74, 0.70, 0.62), "bld_c": (0.50, 0.55, 0.62),
        "bld_d": (0.80, 0.78, 0.76), "bld_e": (0.58, 0.48, 0.42), "roofgrey": (0.40, 0.40, 0.42),
        "helipad": (0.95, 0.75, 0.15),
        "plane_body": (0.96, 0.96, 0.98), "plane_tail": (0.10, 0.25, 0.70),
        "moon": (0.62, 0.62, 0.64), "moon_dark": (0.42, 0.42, 0.45),
        "mercury": (0.36, 0.33, 0.31), "venus": (0.62, 0.44, 0.16), "mars": (0.60, 0.17, 0.08),
        "jupiter": (0.55, 0.33, 0.17), "saturn": (0.58, 0.46, 0.22), "saturn_ring": (0.42, 0.35, 0.22),
        "car_a": (0.85, 0.12, 0.10), "car_b": (0.95, 0.80, 0.10), "car_c": (0.10, 0.35, 0.80),
        "car_d": (0.95, 0.95, 0.95), "car_e": (0.10, 0.10, 0.12), "car_f": (0.15, 0.60, 0.35),
        "car_glass": (0.10, 0.14, 0.20),
    }
    for k, c in spec.items():
        MATS[k] = make_mat(k, c)
    MATS["window"] = make_mat("window", (0.25, 0.55, 0.85), emit=0.35, rough=0.25, spec=0.5)
    MATS["lamp"] = make_mat("lamp", (1.0, 0.92, 0.6), emit=3.0)
    MATS["cloud"] = make_mat("cloud", (0.93, 0.94, 0.96), emit=0.08)
    MATS["sun"] = make_mat("sun", (1.0, 0.72, 0.22), emit=14.0)
    MATS["orbit"] = make_mat("orbit", (0.8, 0.85, 1.0), emit=0.6, alpha=0.5)


# ----------------------------------------------------------------------------- mesh builder
_ICO_CACHE = {}


def _ico(subdiv):
    if subdiv not in _ICO_CACHE:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
        bm.verts.index_update()
        _ICO_CACHE[subdiv] = ([v.co.copy() for v in bm.verts],
                              [[v.index for v in f.verts] for f in bm.faces])
        bm.free()
    return _ICO_CACHE[subdiv]


class Builder:
    """Accumulates flat-shaded polygons with per-face materials into one mesh."""

    def __init__(self):
        self.v, self.f, self.mi, self.mats = [], [], [], []

    def _m(self, name):
        mat = M(name)
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def poly(self, pts, mat):
        i = len(self.v)
        self.v.extend(pts)
        self.f.append(list(range(i, i + len(pts))))
        self.mi.append(self._m(mat))

    def box(self, c, size, mat, rot=0.0, top_mat=None):
        cx, cy, cz = c
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        cr, sr = math.cos(rot), math.sin(rot)
        loc = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
               (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
        i = len(self.v)
        self.v.extend((cx + x * cr - y * sr, cy + x * sr + y * cr, cz + z) for x, y, z in loc)
        m, mt = self._m(mat), self._m(top_mat or mat)
        for idx, mm in (((0, 3, 2, 1), m), ((4, 5, 6, 7), mt), ((0, 1, 5, 4), m),
                        ((1, 2, 6, 5), m), ((2, 3, 7, 6), m), ((3, 0, 4, 7), m)):
            self.f.append([i + k for k in idx])
            self.mi.append(mm)

    def gable(self, c, w, d, z0, rh, mat, rot=0.0):
        """Pitched roof: base w x d at z0, ridge along local Y at z0 + rh."""
        cx, cy = c
        cr, sr = math.cos(rot), math.sin(rot)
        loc = [(-w / 2, -d / 2, z0), (w / 2, -d / 2, z0), (w / 2, d / 2, z0), (-w / 2, d / 2, z0),
               (0, -d / 2, z0 + rh), (0, d / 2, z0 + rh)]
        i = len(self.v)
        self.v.extend((cx + x * cr - y * sr, cy + x * sr + y * cr, z) for x, y, z in loc)
        m = self._m(mat)
        for idx in ((1, 2, 5, 4), (3, 0, 4, 5), (0, 1, 4), (2, 3, 5), (0, 3, 2, 1)):
            self.f.append([i + k for k in idx])
            self.mi.append(m)

    def cyl(self, base, r0, r1, h, n, mat):
        bx, by, bz = base
        i = len(self.v)
        m = self._m(mat)
        ring = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        self.v.extend((bx + r0 * c, by + r0 * s, bz) for c, s in ring)
        if r1 <= 0:
            self.v.append((bx, by, bz + h))
            for k in range(n):
                self.f.append([i + k, i + (k + 1) % n, i + n])
                self.mi.append(m)
        else:
            self.v.extend((bx + r1 * c, by + r1 * s, bz + h) for c, s in ring)
            for k in range(n):
                k2 = (k + 1) % n
                self.f.append([i + k, i + k2, i + n + k2, i + n + k])
                self.mi.append(m)
            self.f.append([i + n + k for k in range(n)])
            self.mi.append(m)

    def ico(self, c, r, mat, scale=(1, 1, 1), subdiv=1, jitter=0.0):
        vs, fs = _ico(subdiv)
        i = len(self.v)
        m = self._m(mat)
        for v in vs:
            j = 1.0 + (random.uniform(-jitter, jitter) if jitter else 0.0)
            self.v.append((c[0] + v.x * r * scale[0] * j, c[1] + v.y * r * scale[1] * j,
                           c[2] + v.z * r * scale[2] * j))
        for f in fs:
            self.f.append([i + k for k in f])
            self.mi.append(m)

    def wall_quad(self, c, normal, w, h, mat):
        """Vertical quad centred at c facing `normal` (a horizontal unit vector)."""
        n = Vector(normal)
        t = Vector((0, 0, 1)).cross(n)
        c = Vector(c)
        z = Vector((0, 0, 1))
        self.poly([tuple(c - t * w / 2 - z * h / 2), tuple(c + t * w / 2 - z * h / 2),
                   tuple(c + t * w / 2 + z * h / 2), tuple(c - t * w / 2 + z * h / 2)], mat)

    def flat_quad(self, x0, y0, x1, y1, z, mat):
        self.poly([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], mat)

    def build(self, name, collection=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        for mat in self.mats:
            me.materials.append(mat)
        me.polygons.foreach_set("material_index", self.mi)
        me.update()
        ob = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(ob)
        return ob


# ----------------------------------------------------------------------------- animation helpers
def bake(idblock, path, values, index=0, first=F_START):
    """One LINEAR key per frame on idblock.path[index]."""
    if idblock.animation_data is None:
        idblock.animation_data_create()
    if idblock.animation_data.action is None:
        idblock.animation_data.action = bpy.data.actions.new(idblock.name + "_act")
    fc = idblock.animation_data.action.fcurves.new(path, index=index)
    fc.keyframe_points.add(len(values))
    co = []
    for i, v in enumerate(values):
        co.extend((first + i, v))
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'
    fc.update()
    return fc


def key_linear(idblock, path, keys, index=0, cycles=False):
    """keys: [(frame, value), ...] with linear interpolation."""
    if idblock.animation_data is None:
        idblock.animation_data_create()
    if idblock.animation_data.action is None:
        idblock.animation_data.action = bpy.data.actions.new(idblock.name + "_act")
    fc = idblock.animation_data.action.fcurves.new(path, index=index)
    for fr, v in keys:
        kp = fc.keyframe_points.insert(fr, v)
        kp.interpolation = 'LINEAR'
    if cycles:
        fc.modifiers.new('CYCLES')
    fc.update()
    return fc


# ----------------------------------------------------------------------------- scene 1+2: street & city
ROADS_Y = [0, -160, 160, -320, 320, -480, 480]     # roads running along Y, at these x
ROADS_X = [-95, 95, -285, 285, -475, 475]          # roads running along X, at these y
MAIN_HW, SIDE_HW, WALK_W = 5.0, 4.0, 2.5


def _gaps(lines, hw):
    """Intervals of [-CITY_HALF, CITY_HALF] not covered by cross roads."""
    edges = sorted(lines)
    out, a = [], -CITY_HALF
    for e in edges:
        out.append((a, e - hw))
        a = e + hw
    out.append((a, CITY_HALF))
    return out


def tree(b, x, y, s=1.0):
    kind = random.random()
    leaf = random.choice(("leaf_a", "leaf_b", "leaf_c"))
    b.cyl((x, y, 0), 0.28 * s, 0.2 * s, 2.4 * s, 6, "trunk")
    if kind < 0.45:   # pine
        b.cyl((x, y, 1.8 * s), 2.3 * s, 0, 3.6 * s, 7, leaf)
        b.cyl((x, y, 4.0 * s), 1.6 * s, 0, 3.2 * s, 7, leaf)
    else:             # round canopy
        b.ico((x, y, 4.3 * s), 2.4 * s, leaf, scale=(1, 1, 1.15), subdiv=1, jitter=0.12)


def house(b, x, y, rot=0.0):
    wall = random.choice(("wall_a", "wall_b", "wall_c", "wall_d"))
    roof = random.choice(("roof_a", "roof_b", "roof_c"))
    w, d, h = random.uniform(8, 10), random.uniform(7.5, 9.5), 3.3
    b.box((x, y, h / 2), (w, d, h), wall, rot)
    b.gable((x, y), w + 1.0, d + 1.0, h, random.uniform(2.2, 3.0), roof, rot)
    cr, sr = math.cos(rot), math.sin(rot)
    ox, oy = w * 0.28, d * 0.2
    b.box((x + ox * cr - oy * sr, y + ox * sr + oy * cr, h + 1.6), (0.8, 0.8, 2.6), "bld_e", rot)


def streetlight(b, x, y, side):
    """side = +1 if the pole is on the +x side of the road (arm reaches toward -x)."""
    b.cyl((x, y, 0), 0.12, 0.08, 7.0, 6, "metal")
    b.box((x - side * 1.25, y, 7.0), (2.6, 0.14, 0.14), "metal")
    b.box((x - side * 2.5, y, 6.93), (0.9, 0.36, 0.14), "lamp")


def hero_tower(b):
    T = TOWER
    cx, cy = (T["x0"] + T["x1"]) / 2, (T["y0"] + T["y1"]) / 2
    sx, sy, h = T["x1"] - T["x0"], T["y1"] - T["y0"], T["h"]
    b.box((cx, cy, h / 2), (sx, sy, h), "tower", top_mat="roofgrey")
    for ex in (T["x0"], T["x1"]):                       # corner trims
        for ey in (T["y0"], T["y1"]):
            b.box((ex, ey, h / 2), (1.2, 1.2, h + 1.0), "tower_trim")
    floors = int((h - 8) // 4)
    for normal, face_c, width in (((-1, 0, 0), (T["x0"], cy), sy), ((1, 0, 0), (T["x1"], cy), sy),
                                  ((0, -1, 0), (cx, T["y0"]), sx), ((0, 1, 0), (cx, T["y1"]), sx)):
        n = Vector(normal)
        t = Vector((0, 0, 1)).cross(n)
        cols = int((width - 3) // 4)
        for fl in range(floors):
            z = 6 + fl * 4
            for c in range(cols):
                off = (c - (cols - 1) / 2) * 4.0
                p = Vector((face_c[0], face_c[1], z)) + t * off + n * 0.08
                b.wall_quad(p, n, 2.8, 2.4, "window")
    # roof furniture, seen once the camera clears the top
    b.box((cx - 6, cy + 8, h + 2.5), (10, 8, 5), "tower_trim")
    b.cyl((cx + 6, cy - 6, h), 7.0, 7.0, 0.4, 16, "helipad")
    b.cyl((cx + 6, cy - 6, h + 0.4), 5.6, 5.6, 0.1, 16, "tower_trim")
    b.box((cx + 6, cy - 6, h + 0.6), (0.9, 5.0, 0.1), "paint")
    b.box((cx + 4.2, cy - 6, h + 0.6), (0.9, 5.0, 0.1), "paint")
    b.box((cx + 7.8, cy - 6, h + 0.6), (0.9, 5.0, 0.1), "paint")
    b.box((cx + 6, cy - 6, h + 0.62), (3.6, 0.9, 0.1), "paint")
    b.cyl((cx - 10, cy - 10, h), 0.5, 0.15, 28, 6, "metal")


def build_city():
    b = Builder()
    H = CITY_HALF
    b.flat_quad(-H, -H, H, H, 0.0, "grass")

    # roads, sidewalks, lane paint
    for x in ROADS_Y:
        hw = MAIN_HW if x == 0 else SIDE_HW
        b.flat_quad(x - hw, -H, x + hw, H, 0.04, "asphalt")
        for a, c in _gaps(ROADS_X, SIDE_HW + WALK_W):
            for s in (-1, 1):
                xc = x + s * (hw + WALK_W / 2)
                b.box((xc, (a + c) / 2, 0.08), (WALK_W, c - a, 0.16), "sidewalk")
        step = 6.0 if x == 0 else 14.0
        y = -H
        while y < H:
            if not (x == 0 and 2.0 < y + step * 0.25 < 8.0):   # leave room for the crosswalk
                b.flat_quad(x - 0.09, y, x + 0.09, y + step * 0.5, 0.07, "paint")
            y += step
    for y in ROADS_X:
        b.flat_quad(-H, y - SIDE_HW, H, y + SIDE_HW, 0.03, "asphalt")
        for a, c in _gaps(ROADS_Y, MAIN_HW + WALK_W):
            for s in (-1, 1):
                yc = y + s * (SIDE_HW + WALK_W / 2)
                b.box(((a + c) / 2, yc, 0.08), (c - a, WALK_W, 0.16), "sidewalk")
    # zebra crossing just ahead of the viewer, plus street furniture underfoot
    x = -MAIN_HW + 0.45
    while x < MAIN_HW - 0.3:
        b.flat_quad(x, 3.0, x + 0.45, 6.4, 0.07, "paint")
        x += 0.9
    b.flat_quad(-MAIN_HW + 0.3, 1.9, MAIN_HW - 0.3, 2.25, 0.07, "paint")          # stop line
    b.cyl((-0.95, -0.35, 0.04), 0.36, 0.36, 0.035, 12, "metal")                    # manhole cover
    b.cyl((-0.95, -0.35, 0.075), 0.27, 0.27, 0.01, 12, "manhole")
    b.flat_quad(1.25, -0.9, 1.45, 0.2, 0.07, "paint")                              # lane arrow
    b.poly([(1.0, 0.2, 0.07), (1.7, 0.2, 0.07), (1.35, 0.85, 0.07)], "paint")
    for _ in range(60):                                                            # tarmac patches
        px, py = random.uniform(-MAIN_HW + 0.3, MAIN_HW - 0.3), random.uniform(-14, 14)
        w, d = random.uniform(0.25, 1.1), random.uniform(0.25, 1.4)
        b.flat_quad(px - w / 2, py - d / 2, px + w / 2, py + d / 2, 0.05,
                    random.choice(("asphalt_dark", "asphalt_light")))

    hero_tower(b)
    T = TOWER
    occupied = [(T["x0"] - 4, T["y0"] - 4, T["x1"] + 4, T["y1"] + 4)]

    # the home street around the origin
    for side in (-1, 1):
        for row, xr in enumerate((16.0, 44.0)):
            y = -78.0
            while y <= 78.0:
                hx = side * xr
                if not any(r[0] - 6 < hx < r[2] + 6 and r[1] - 6 < y < r[3] + 6 for r in occupied[:1]):
                    house(b, hx, y + random.uniform(-1, 1), rot=random.choice((0, math.pi / 2)))
                    occupied.append((hx - 6, y - 6, hx + 6, y + 6))
                y += 17.0
    for k in range(-18, 19):
        y = 4.0 + 16.0 * k
        if abs(y) < 300 and all(abs(y - ry) > SIDE_HW + 4 for ry in ROADS_X):
            side = 1 if k % 2 else -1
            streetlight(b, side * (MAIN_HW + 0.8), y, side)
    for tx, ty, ts in ((-9.6, -2.8, 1.0), (-9.9, 8.5, 1.2), (8.3, -15.5, 1.1), (-9.4, -17.0, 0.9),
                       (-10.2, 21.0, 1.15), (8.2, -29.0, 1.0)):
        tree(b, tx, ty, ts)                              # verge trees the camera rises past
    placed = 0
    while placed < 110:
        x, y = random.uniform(-62, 62), random.uniform(-88, 88)
        if abs(x) < MAIN_HW + WALK_W + 1.5:
            continue
        if any(r[0] < x < r[2] and r[1] < y < r[3] for r in occupied):
            continue
        tree(b, x, y, random.uniform(0.8, 1.35))
        placed += 1

    # the rest of the city: blocks of boxes, taller near the tower
    xs = sorted(ROADS_Y) + [H + 40]
    ys = sorted(ROADS_X) + [H + 40]
    xs, ys = [-H - 40] + xs, [-H - 40] + ys
    palette = ("bld_a", "bld_b", "bld_c", "bld_d", "bld_e")
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1 = max(xs[i] + 9, -H + 4), min(xs[i + 1] - 9, H - 4)
            y0, y1 = max(ys[j] + 8, -H + 4), min(ys[j + 1] - 8, H - 4)
            nx, ny = max(int((x1 - x0) // 30), 1), max(int((y1 - y0) // 30), 1)
            lx, ly = (x1 - x0) / nx, (y1 - y0) / ny
            for a in range(nx):
                for c in range(ny):
                    cx, cy = x0 + (a + 0.5) * lx, y0 + (c + 0.5) * ly
                    if abs(cx) < 64 and abs(cy) < 92:
                        continue                         # home street, already built
                    d = math.hypot(cx - 25, cy - 8)
                    if random.random() < 0.10 + 0.25 * min(d / 700, 1):
                        for _ in range(random.randint(2, 5)):
                            tree(b, cx + random.uniform(-lx / 3, lx / 3),
                                 cy + random.uniform(-ly / 3, ly / 3), random.uniform(0.9, 1.6))
                        continue
                    h = max((150 * math.exp(-d / 210) + 9) * random.uniform(0.45, 1.3), 7.0)
                    w, dd = lx - random.uniform(5, 10), ly - random.uniform(5, 10)
                    mat = random.choice(palette)
                    b.box((cx, cy, h / 2), (w, dd, h), mat, top_mat="roofgrey")
                    if h > 26:
                        b.box((cx + w * 0.15, cy - dd * 0.15, h + 1.5), (w * 0.35, dd * 0.3, 3.0), "roofgrey")
                        z = 5.0
                        while z < h - 2:
                            for nrm, fc_, wd in (((-1, 0, 0), (cx - w / 2, cy), dd), ((1, 0, 0), (cx + w / 2, cy), dd),
                                                 ((0, -1, 0), (cx, cy - dd / 2), w), ((0, 1, 0), (cx, cy + dd / 2), w)):
                                n = Vector(nrm)
                                b.wall_quad(Vector((fc_[0], fc_[1], z)) + n * 0.06, n, wd - 2.0, 1.7, "window")
                            z += 4.0
    return b.build("City")


def build_cars():
    cars = []
    period = int(2 * CITY_HALF / 12.0 * FPS)             # frames to cross the city at 12 m/s
    colors = ("car_a", "car_b", "car_c", "car_d", "car_e", "car_f")

    def car(name):
        b = Builder()
        col = random.choice(colors)
        b.box((0, 0, 0.62), (1.85, 4.4, 0.75), col)
        b.box((0, -0.25, 1.25), (1.6, 2.3, 0.55), "car_glass", top_mat=col)
        return b.build(name)

    # main road: +Y traffic at x=+2.4, -Y traffic at x=-2.4; none under the camera before ~5 s
    t_cross = {1: [5.5, 12, 19, 27, 36, 46, 58, 71, 84, 95],
               -1: [8, 15, 23, 31, 40, 52, 64, 77, 90, 98]}
    n = 0
    for direction, times in t_cross.items():
        for t in times:
            ob = car(f"Car{n:02d}")
            n += 1
            ob.location = (2.4 * direction, 0, 0.05)
            ob.rotation_euler[2] = 0 if direction > 0 else math.pi
            f0 = frame_of(t) - period // 2
            key_linear(ob, "location", [(f0, -CITY_HALF * direction), (f0 + period, CITY_HALF * direction)],
                       index=1, cycles=True)
            cars.append(ob)
    # cross streets
    for y in ROADS_X[:4]:
        for direction in (1, -1):
            for k in range(3):
                ob = car(f"Car{n:02d}")
                n += 1
                ob.location = (0, y - 2.0 * direction, 0.05)
                ob.rotation_euler[2] = -math.pi / 2 * direction
                f0 = -random.randint(0, period)
                key_linear(ob, "location", [(f0, -CITY_HALF * direction), (f0 + period, CITY_HALF * direction)],
                           index=0, cycles=True)
                cars.append(ob)
    return cars


# ----------------------------------------------------------------------------- scene 3: land, clouds, plane
def build_countryside():
    """A patchwork of fields, woods and lakes draped on the Earth sphere around the city."""
    b = Builder()
    cell, reach = 420.0, 13_000.0
    n = int(reach // cell)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
            if math.hypot(cx, cy) > reach or (abs(cx) < CITY_HALF + cell / 2 and abs(cy) < CITY_HALF + cell / 2):
                continue
            wood = noise.noise(Vector((cx / 2600.0, cy / 2600.0, 0.3)))
            water = noise.noise(Vector((cx / 1900.0 + 5.0, cy / 1900.0, 2.2)))
            if water > 0.46:
                mat, inset = "lake", 0.0
            elif wood < -0.2:
                mat, inset = "forest", 0.0
            else:
                mat, inset = random.choice(("field_a", "field_b", "field_c", "field_a", "grass")), 14.0
            h = cell / 2 - inset
            pts = [(x, y, on_earth(x, y, 24.0)) for x, y in
                   ((cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h))]
            b.poly(pts, mat)
    # country roads continuing out of the city
    for ang in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        prev = None
        for k in range(40):
            r = CITY_HALF - 10 + k * 230
            x, y = r * math.cos(ang), r * math.sin(ang)
            px, py = -math.sin(ang) * 7, math.cos(ang) * 7
            cur = ((x - px, y - py, on_earth(x - px, y - py, 30)), (x + px, y + py, on_earth(x + px, y + py, 30)))
            if prev:
                b.poly([prev[0], prev[1], cur[1], cur[0]], "asphalt")
            prev = cur
    return b.build("Countryside")


def build_clouds():
    b = Builder()

    def cluster(x, y, h, size):
        for _ in range(random.randint(6, 10)):
            ox, oy = random.uniform(-size, size) * 1.1, random.uniform(-size, size) * 0.7
            r = size * random.uniform(0.3, 0.8)
            px, py = x + ox, y + oy
            b.ico((px, py, on_earth(px, py, h) + random.uniform(-0.1, 0.1) * size), r, "cloud",
                  scale=(1, 1, 0.42), subdiv=1, jitter=0.08)

    # local layer the camera punches through (kept clear of the Z axis itself)
    for _ in range(170):
        r = math.sqrt(random.uniform(0.004, 1.0)) * 15000
        a = random.uniform(0, 2 * math.pi)
        cluster(r * math.cos(a), r * math.sin(a), random.uniform(CLOUD_BASE + 300, CLOUD_TOP - 300),
                random.uniform(260, 650) * (1 + r / 9000))
    ob_local = b.build("CloudsLocal")

    # weather systems over the rest of the globe
    b = Builder()
    n = 0
    while n < 230:
        d = Vector((random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))).normalized()
        if d.z < -0.35 or (d.z > 0.92):
            continue
        size = random.uniform(1400, 4200)
        c = d * (EARTH_R + 2600)
        for _ in range(random.randint(4, 8)):
            off = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))) * size * 1.3
            p = (c + off).normalized() * (EARTH_R + 2600)
            sc = size * random.uniform(0.4, 0.9)
            b.ico((p.x, p.y, p.z - EARTH_R - EARTH_SINK), sc, "cloud", scale=(1, 1, 1), subdiv=1, jitter=0.1)
        n += 1
    ob_global = b.build("CloudsGlobal")
    # squash global puffs radially: cheap trick = scale the puff verts toward the shell
    me = ob_global.data
    centre = Vector((0, 0, -EARTH_R - EARTH_SINK))
    for v in me.vertices:
        rel = v.co - centre
        rr = rel.length
        v.co = centre + rel * ((EARTH_R + 2600 + (rr - EARTH_R - 2600) * 0.3) / rr)
    return ob_local, ob_global


def build_cloud_volume():
    """A real volumetric slab on the Z axis so the camera whites out as it breaches."""
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, (CLOUD_BASE + CLOUD_TOP) / 2))
    ob = bpy.context.object
    ob.name = "CloudVolume"
    ob.scale = (4000, 4000, (CLOUD_TOP - CLOUD_BASE) / 2)
    m = bpy.data.materials.new("cloud_volume")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 3.2
    nz.inputs["Detail"].default_value = 4.0
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.47, 0.62
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.0, 1.0
    flat = nt.nodes.new("ShaderNodeVectorMath")
    flat.operation = 'MULTIPLY'
    flat.inputs[1].default_value = (1, 1, 0)
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = 'LENGTH'
    fall = nt.nodes.new("ShaderNodeMapRange")
    fall.inputs["From Min"].default_value, fall.inputs["From Max"].default_value = 0.30, 0.95
    core = nt.nodes.new("ShaderNodeMapRange")          # solid puff right on the Z axis
    core.inputs["From Min"].default_value, core.inputs["From Max"].default_value = 0.10, 0.24
    core.inputs["To Min"].default_value, core.inputs["To Max"].default_value = 1.0, 0.0
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = 'MAXIMUM'
    fall.inputs["To Min"].default_value, fall.inputs["To Max"].default_value = 1.0, 0.0
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    strength = nt.nodes.new("ShaderNodeValue")
    strength.name = "Strength"
    strength.outputs[0].default_value = 0.012
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = 'MULTIPLY'
    L = nt.links.new
    L(tc.outputs["Object"], nz.inputs["Vector"])
    L(nz.outputs["Fac"], mr.inputs["Value"])
    L(tc.outputs["Object"], flat.inputs[0])
    L(flat.outputs["Vector"], ln.inputs[0])
    L(ln.outputs["Value"], fall.inputs["Value"])
    L(mr.outputs["Result"], mul.inputs[0])
    L(fall.outputs["Result"], mul.inputs[1])
    L(ln.outputs["Value"], core.inputs["Value"])
    L(mul.outputs[0], mx.inputs[0])
    L(core.outputs["Result"], mx.inputs[1])
    L(mx.outputs[0], mul2.inputs[0])
    L(strength.outputs[0], mul2.inputs[1])
    L(mul2.outputs[0], vol.inputs["Density"])
    glow = nt.nodes.new("ShaderNodeMath")              # emission tied to density -> bright, not grey
    glow.operation = 'MULTIPLY'
    glow.inputs[1].default_value = 0.8
    L(mul2.outputs[0], glow.inputs[0])
    L(glow.outputs[0], vol.inputs["Emission Strength"])
    vol.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    vol.inputs["Color"].default_value = (1, 1, 1, 1)
    vol.inputs["Anisotropy"].default_value = 0.2
    L(vol.outputs["Volume"], out.inputs["Volume"])
    ob.data.materials.append(m)
    # thin the slab out once it is just a speck far below
    for t, v in ((36.0, 0.012), (39.5, 0.0)):
        strength.outputs[0].default_value = v
        strength.outputs[0].keyframe_insert("default_value", frame=frame_of(t))
    # only render the slab while the camera is near, in or just above it (a volume left in the
    # scene once clip_start passes volumetric_end turns EEVEE frames black)
    for fr, hidden in ((1, True), (frame_of(28.0) - 1, True), (frame_of(28.0), False),
                       (frame_of(39.5), False), (frame_of(39.5) + 1, True)):
        ob.hide_render = hidden
        ob.keyframe_insert("hide_render", frame=fr)
    return ob


def build_airplane():
    b = Builder()
    b.box((0, 0, 0), (58, 6, 6), "plane_body")
    b.box((31, 0, -0.5), (6, 4.4, 4.4), "plane_body")          # nose
    b.box((0, 0, -0.8), (13, 112, 1.2), "plane_body")           # wings
    b.box((-25, 0, 1.2), (7, 38, 1.0), "plane_body")            # tailplane
    b.box((-25, 0, 7.0), (10, 1.2, 11), "plane_tail")           # fin
    for s in (-1, 1):
        b.box((3, s * 24, -3.2), (9, 3.4, 3.4), "plane_tail")   # engines
    ob = b.build("Airplane")
    alt = CLOUD_TOP + 500
    a, c = Vector((-2600, -1200, alt)), Vector((2400, 1300, alt))
    ob.rotation_euler[2] = math.atan2(c.y - a.y, c.x - a.x)
    f0, f1 = frame_of(33), frame_of(41)
    for i in range(3):
        key_linear(ob, "location", [(f0, a[i]), (f1, c[i])], index=i)
    return ob


# ----------------------------------------------------------------------------- scene 4+5: space
def build_earth():
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=5, radius=EARTH_R)
    me = bpy.data.meshes.new("Earth")
    bm.to_mesh(me)
    bm.free()
    names = ("grass", "forest", "desert", "ocean", "ice")
    for n in names:
        me.materials.append(M(n))
    idx = []
    for p in me.polygons:
        d = p.center.normalized()
        v = noise.noise(d * 1.7 + Vector((3.1, 1.7, 9.2))) + 0.45 * noise.noise(d * 4.3) \
            + 0.2 * noise.noise(d * 9.0)
        ang = math.degrees(math.acos(max(min(d.z, 1), -1)))
        v += max(0.0, 1.0 - ang / 24.0) * 1.6           # make sure the city stands on land
        lat = abs(d.y)                                   # poles on the +/-Y axis: top/bottom of frame
        if lat > 0.90 + 0.05 * noise.noise(d * 6.0):
            idx.append(4)
        elif v < 0.02:
            idx.append(3)
        else:
            w = noise.noise(d * 3.0 + Vector((7, 7, 7)))
            idx.append(2 if (w > 0.28 and lat < 0.55 and ang > 18) else (1 if w < -0.12 and ang > 6 else 0))
    me.polygons.foreach_set("material_index", idx)
    me.update()
    ob = bpy.data.objects.new("Earth", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = (0, 0, -EARTH_R - EARTH_SINK)

    # thin blue rim of atmosphere
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=EARTH_R + 1400, location=ob.location)
    atm = bpy.context.object
    atm.name = "Atmosphere"
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new("atmosphere")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.35, 0.65, 1.0, 1)
    em.inputs["Strength"].default_value = 1.6
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = 'POWER'
    pw.inputs[1].default_value = 3.0
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.blend_method = 'BLEND'
    m.shadow_method = 'NONE'
    m.use_backface_culling = True
    atm.data.materials.append(m)
    return ob, atm


def build_moon():
    b = Builder()
    vs, fs = _ico(3)
    r = 9000.0
    i0 = len(b.v)
    b.v.extend(tuple(v * r) for v in vs)
    for f in fs:
        c = (vs[f[0]] + vs[f[1]] + vs[f[2]]) / 3
        b.f.append([i0 + k for k in f])
        b.mi.append(b._m("moon_dark" if noise.noise(c * 2.6) > 0.18 else "moon"))
    ob = b.build("Moon")
    a, c = Vector((150_000, 45_000, 8_000)), Vector((45_000, 20_000, 8_000))
    for i in range(3):
        key_linear(ob, "location", [(frame_of(40), a[i]), (frame_of(52), c[i])], index=i)
    key_linear(ob, "rotation_euler", [(frame_of(40), 0.0), (frame_of(60), 1.2)], index=2)
    return ob


def build_solar_system():
    sc = bpy.context.scene.collection
    b = Builder()
    b.ico((0, 0, 0), SUN_R, "sun", subdiv=3)
    sun = b.build("Sun")
    sun.location = SUN_POS

    orbit_mat = M("orbit")
    bsdf = orbit_mat.node_tree.nodes["Principled BSDF"]
    for t, a, e in ((54.5, 0.0, 0.0), (59.0, 0.3, 0.6)):     # rings fade in as they get thin
        bsdf.inputs["Alpha"].default_value = a
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=frame_of(t))
        bsdf.inputs["Emission Strength"].default_value = e
        bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=frame_of(t))

    #        name      orbit km  radius km  start angle  material
    planets = [("Mercury", 450, 35, 70, "mercury"), ("Venus", 800, 60, 300, "venus"),
               ("EarthOrbit", 1200, 0, 0, None), ("Mars", 1800, 50, -42, "mars"),
               ("Jupiter", 3000, 150, 12, "jupiter"), ("Saturn", 4500, 120, -22, "saturn")]
    made = [sun]
    for name, orbit_km, r_km, ang, mat in planets:
        orbit = orbit_km * 1000.0
        rb = Builder()                                    # flat annulus (the camera only looks down)
        seg, half_w = 220, 4500.0
        for k in range(seg):
            a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
            ri, ro = orbit - half_w, orbit + half_w
            rb.poly([(ri * math.cos(a0), ri * math.sin(a0), 0), (ro * math.cos(a0), ro * math.sin(a0), 0),
                     (ro * math.cos(a1), ro * math.sin(a1), 0), (ri * math.cos(a1), ri * math.sin(a1), 0)],
                    "orbit")
        ring = rb.build(f"Orbit_{name}")
        ring.location = SUN_POS
        made.append(ring)
        if not mat:
            continue
        pivot = bpy.data.objects.new(f"Pivot_{name}", None)
        sc.objects.link(pivot)
        pivot.location = SUN_POS
        pb = Builder()
        pb.ico((0, 0, 0), r_km * 1000.0, mat, subdiv=3)
        if name == "Saturn":
            for k in range(28):                               # flat ring of slabs
                a0, a1 = 2 * math.pi * k / 28, 2 * math.pi * (k + 1) / 28
                ri, ro = r_km * 1500.0, r_km * 2300.0
                pb.poly([(ri * math.cos(a0), ri * math.sin(a0), 0), (ro * math.cos(a0), ro * math.sin(a0), 0),
                         (ro * math.cos(a1), ro * math.sin(a1), 0), (ri * math.cos(a1), ri * math.sin(a1), 0)],
                        "saturn_ring")
        p = pb.build(name)
        p.parent = pivot
        p.location = (orbit, 0, 0)
        a0 = math.radians(ang)
        speed = math.radians(26.0) * (1200.0 / orbit_km) ** 1.5      # inner planets orbit faster
        key_linear(pivot, "rotation_euler", [(frame_of(48), a0), (frame_of(60), a0 + speed)], index=2)
        made.append(p)
    return made


# ----------------------------------------------------------------------------- world, light, camera
def build_world():
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    lp = nt.nodes.new("ShaderNodeLightPath")
    amb = nt.nodes.new("ShaderNodeBackground")        # what lights the scene (constant soft fill)
    amb.inputs["Color"].default_value = (0.62, 0.76, 1.0, 1)
    amb.inputs["Strength"].default_value = 0.5
    cam_bg = nt.nodes.new("ShaderNodeBackground")     # what the camera sees: blue sky -> starfield
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 180.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (1, 1, 1, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.07, (0, 0, 0, 1)
    sky = nt.nodes.new("ShaderNodeMixRGB")
    sky.name = "SkyFade"
    sky.inputs["Color1"].default_value = (0.30, 0.56, 0.95, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    L = nt.links.new
    L(tc.outputs["Generated"], vor.inputs["Vector"])
    L(vor.outputs["Distance"], ramp.inputs["Fac"])
    L(ramp.outputs["Color"], sky.inputs["Color2"])
    L(sky.outputs["Color"], cam_bg.inputs["Color"])
    L(lp.outputs["Is Camera Ray"], mix.inputs[0])
    L(amb.outputs[0], mix.inputs[1])
    L(cam_bg.outputs[0], mix.inputs[2])
    L(mix.outputs[0], out.inputs["Surface"])
    for t, v in ((38.5, 0.0), (46.0, 1.0)):            # the sky fades to black space
        sky.inputs["Fac"].default_value = v
        sky.inputs["Fac"].keyframe_insert("default_value", frame=frame_of(t))
    return w


def build_light_and_camera():
    scene = bpy.context.scene
    ld = bpy.data.lights.new("SunLamp", 'SUN')
    ld.energy = 2.9
    ld.angle = math.radians(1.5)
    ld.color = (1.0, 0.96, 0.88)
    ld.shadow_cascade_count = 4
    lo = bpy.data.objects.new("SunLamp", ld)
    scene.collection.objects.link(lo)
    lo.rotation_euler = SUN_DIR.to_track_quat('Z', 'Y').to_euler()

    cd = bpy.data.cameras.new("POV")
    cd.lens = 18.0
    cd.sensor_width = 36.0
    cam = bpy.data.objects.new("POV", cd)
    scene.collection.objects.link(cam)
    cam.rotation_euler = (0, 0, 0)                     # identity = looking straight down -Z
    cam.location = (0, 0, ALTITUDE_KNOTS[0][1])
    scene.camera = cam

    frames = range(F_START, F_END + 1)
    z = [altitude(f) for f in frames]
    bake(cam, "location", z, index=2)                  # the only camera motion: straight up
    # keep depth precision sane across 7 orders of magnitude
    bake(cd, "clip_start", [min(max(0.02 * v, 0.05), 50_000.0) for v in z])
    bake(cd, "clip_end", [max(40.0 * v, 3000.0) for v in z])
    bake(ld, "shadow_cascade_max_distance", [min(max(4.0 * v, 40.0), 6000.0) for v in z])
    return cam, lo


# ----------------------------------------------------------------------------- render setup
def setup_render(args):
    scene = bpy.context.scene
    r = scene.render
    r.engine = args.engine
    r.resolution_x, r.resolution_y, r.resolution_percentage = 1920, 1080, args.pct
    r.fps = FPS
    scene.frame_start, scene.frame_end = F_START, F_END
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.compression = 30
    r.use_overwrite = False                            # lets an interrupted render resume
    r.use_placeholder = False
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    if args.engine == 'CYCLES':
        scene.cycles.samples = max(args.samples, 24)
        scene.cycles.use_denoising = True
    else:
        e = scene.eevee
        e.taa_render_samples = args.samples
        e.use_soft_shadows = True
        e.shadow_cascade_size = '2048'
        e.use_gtao = True
        e.gtao_distance = 2.5
        e.use_bloom = True
        e.bloom_threshold = 2.2
        e.bloom_intensity = 0.25
        e.bloom_radius = 6.0
        e.volumetric_start = 1.0
        e.volumetric_end = 20_000.0    # constant: animating this setting is ignored at render time
        e.volumetric_tile_size = '8'
        e.volumetric_samples = 64
        e.use_volumetric_lights = True
        e.use_volumetric_shadows = False


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_materials()
    build_world()
    build_city()
    build_cars()
    build_countryside()
    build_clouds()
    if not args.no_volume:
        build_cloud_volume()
    build_airplane()
    build_earth()
    build_moon()
    build_solar_system()
    build_light_and_camera()
    setup_render(args)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(here, "renders"))
    p.add_argument("--stills", default="")
    p.add_argument("--animation", action="store_true")
    p.add_argument("--start", type=int, default=F_START)
    p.add_argument("--end", type=int, default=F_END)
    p.add_argument("--pct", type=int, default=100)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--engine", default="BLENDER_EEVEE")
    p.add_argument("--no-volume", action="store_true")
    p.add_argument("--save-blend", default="")
    return p.parse_args(argv)


def main():
    import time
    args = parse_args()
    build_scene(args)
    scene = bpy.context.scene
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    if args.stills:
        os.makedirs(os.path.join(args.out, "previews"), exist_ok=True)
        for f in (int(s) for s in args.stills.split(",")):
            f = min(max(f, F_START), F_END)
            scene.frame_set(f)
            scene.render.filepath = os.path.join(args.out, "previews", f"frame_{f:04d}.png")
            scene.render.use_overwrite = True
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"[grow_film] still {f}: t={(f - 1) / FPS:5.2f}s alt={altitude(f):,.1f} m "
                  f"render={time.time() - t0:.1f}s", flush=True)
    if args.animation:
        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
        scene.frame_start, scene.frame_end = args.start, args.end
        scene.render.filepath = os.path.join(args.out, "frames", "f_")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
