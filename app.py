import os
import math
import sys
import re
import json
import base64
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import cadquery as cq
from groq import Groq

# ================== ENV ==================
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cq_gears"))
from cq_gears import SpurGear, RingGear, BevelGear

# ================== APP ==================
app = Flask(__name__)
CORS(app, origins="*")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/")
def home():
    return "Backend running"

# ================== SECURITY ==================
SAFE_GLOBALS = {
    "__builtins__": {},
    "cq": cq,
    "math": math,
    "SpurGear": SpurGear,
    "RingGear": RingGear,
    "BevelGear": BevelGear,
}

BLOCKED_PATTERNS = [
    "import os", "import sys", "open(", "__import__", "eval(", "exec(",
    "subprocess", "socket", "shutil", "pickle", "builtins", "globals()",
    "locals()", "getattr", "setattr", "delattr", "compile(",
]

def is_code_safe(code: str):
    code_lower = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in code_lower:
            return False, pattern
    return True, None

# ================== LIMITS ==================
MAX_DIM   = 2000   # mm
MAX_TEETH = 300

def enforce_limits(params: dict):
    for k, v in params.items():
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError(f"'{k}' must be >= 0, got {v}")
            if k not in ("teeth", "grooves", "holes", "plates", "turns", "steps") and v > MAX_DIM:
                raise ValueError(f"'{k}' exceeds max limit ({MAX_DIM} mm)")
        if k == "teeth" and v > MAX_TEETH:
            raise ValueError(f"Too many teeth (max {MAX_TEETH})")

# ================== VALIDATORS ==================
def validate_positive(*args):
    for name, val in args:
        if val <= 0:
            raise ValueError(f"'{name}' must be > 0, got {val}")

def validate_hollow(od, id_, label="Part"):
    if id_ >= od:
        raise ValueError(f"{label}: inner diameter ({id_}mm) must be less than outer diameter ({od}mm)")

def validate_gear_bore(pitch_dia, bore, label="Gear"):
    if bore > 0 and bore >= pitch_dia:
        raise ValueError(f"{label}: bore ({bore}mm) must be smaller than pitch diameter ({pitch_dia}mm)")

def validate_flange_holes(od, pcd, hole_dia):
    if pcd / 2 + hole_dia / 2 >= od / 2:
        raise ValueError("Bolt holes extend beyond flange OD. Reduce PCD or hole diameter.")

# ================== TEMPLATE MAKERS ==================

def make_shaft(length, diameter, chamfer=0, fillet=0):
    validate_positive(("length", length), ("diameter", diameter))
    r = cq.Workplane("XY").circle(diameter / 2).extrude(length)
    if chamfer > 0:
        r = r.edges("|Z").chamfer(chamfer)
    if fillet > 0:
        try:
            r = r.edges("not |Z").fillet(fillet)
        except Exception:
            pass
    return r

def make_pulley(od, bore, width, grooves=1, chamfer=0):
    validate_positive(("od", od), ("bore", bore), ("width", width))
    if bore >= od:
        raise ValueError(f"Bore ({bore}mm) must be smaller than OD ({od}mm)")
    r = cq.Workplane("XY").circle(od / 2).extrude(width)
    r = r.faces(">Z").workplane().circle(bore / 2).cutThruAll()
    # Simple V-groove per groove
    groove_width = min(8, width / max(grooves, 1) * 0.6)
    groove_depth = min(10, (od - bore) / 4)
    for i in range(int(grooves)):
        z_pos = (i + 0.5) * (width / grooves)
        r = (r.workplane(offset=z_pos - groove_width / 2)
              .circle(od / 2).workplane(offset=groove_width)
              .circle(od / 2 - groove_depth).loft(combine="cut"))
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_coupling(bore_a, bore_b, length, od, chamfer=0):
    validate_positive(("bore_a", bore_a), ("bore_b", bore_b), ("length", length), ("od", od))
    if bore_a >= od or bore_b >= od:
        raise ValueError("Bore must be smaller than OD")
    half = length / 2
    r = cq.Workplane("XY").circle(od / 2).extrude(length)
    # Bore A (bottom half)
    r = r.faces("<Z").workplane().circle(bore_a / 2).extrude(half, combine="cut")
    # Bore B (top half)
    r = r.faces(">Z").workplane().circle(bore_b / 2).extrude(half, combine="cut")
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_axle(length, diameter, shoulder_dia=0, chamfer=0):
    validate_positive(("length", length), ("diameter", diameter))
    r = cq.Workplane("XY").circle(diameter / 2).extrude(length)
    if shoulder_dia > diameter:
        shoulder_h = min(length * 0.1, 20)
        shoulder = cq.Workplane("XY").circle(shoulder_dia / 2).extrude(shoulder_h)
        r = r.union(shoulder)
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_nut(across_flats, height, hole_dia, chamfer=0):
    validate_positive(("across_flats", across_flats), ("height", height), ("hole_dia", hole_dia))
    if hole_dia >= across_flats:
        raise ValueError(f"Hole diameter ({hole_dia}mm) must be smaller than across-flats ({across_flats}mm)")
    r = cq.Workplane("XY").polygon(6, across_flats / math.sqrt(3)).extrude(height)
    r = r.faces(">Z").workplane().circle(hole_dia / 2).cutThruAll()
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
            r = r.faces("<Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_bolt(thread_dia, length, head_height, across_flats, chamfer=0):
    """Simplified bolt: hex head + cylindrical shank."""
    validate_positive(("thread_dia", thread_dia), ("length", length))
    head_od = across_flats / math.sqrt(3) if across_flats > 0 else thread_dia * 1.5
    head_h  = head_height if head_height > 0 else thread_dia * 0.7
    head    = cq.Workplane("XY").polygon(6, head_od).extrude(head_h)
    shank   = cq.Workplane("XY").workplane(offset=head_h).circle(thread_dia / 2).extrude(length)
    r       = head.union(shank)
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_washer(od, id_, thickness, chamfer=0):
    validate_positive(("od", od), ("id_", id_), ("thickness", thickness))
    validate_hollow(od, id_, "Washer")
    r = cq.Workplane("XY").circle(od / 2).extrude(thickness)
    r = r.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_pin(diameter, length, chamfer=0):
    validate_positive(("diameter", diameter), ("length", length))
    r = cq.Workplane("XY").circle(diameter / 2).extrude(length)
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
            r = r.faces("<Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_spur_gear(module, teeth, face_width, bore=0, chamfer=0):
    validate_positive(("module", module), ("teeth", teeth), ("face_width", face_width))
    pitch_dia = module * teeth
    if bore > 0:
        validate_gear_bore(pitch_dia, bore, "Spur Gear")
        gear = SpurGear(module=module, teeth_number=int(teeth), width=face_width, bore_d=bore)
    else:
        gear = SpurGear(module=module, teeth_number=int(teeth), width=face_width)
    r = gear.build()
    return r

def make_bevel_gear(module, teeth, face_width, bore=0, pitch_angle=45):
    validate_positive(("module", module), ("teeth", teeth), ("face_width", face_width))
    gear = BevelGear(module=module, teeth_number=int(teeth), width=face_width)
    return gear.build()

def make_ring_gear(module, teeth, face_width):
    validate_positive(("module", module), ("teeth", teeth), ("face_width", face_width))
    gear = RingGear(module=module, teeth_number=int(teeth), width=face_width)
    return gear.build()

def make_helical_gear(module, teeth, face_width, bore=0, helix_angle=20):
    """Approximated as spur gear — true helical requires cq_gears HeliGear if available."""
    validate_positive(("module", module), ("teeth", teeth), ("face_width", face_width))
    pitch_dia = module * teeth
    if bore > 0:
        validate_gear_bore(pitch_dia, bore, "Helical Gear")
        gear = SpurGear(module=module, teeth_number=int(teeth), width=face_width, bore_d=bore)
    else:
        gear = SpurGear(module=module, teeth_number=int(teeth), width=face_width)
    return gear.build()

def make_worm_gear(module, teeth, face_width, bore=0):
    """Approximated as spur gear with appropriate proportions."""
    return make_helical_gear(module, teeth, face_width, bore, helix_angle=20)

def make_clutch(od, id_, plate_thick, plates=3):
    validate_positive(("od", od), ("id_", id_), ("plate_thick", plate_thick))
    validate_hollow(od, id_, "Clutch")
    total_h = plate_thick * int(plates)
    r = cq.Workplane("XY").circle(od / 2).extrude(total_h)
    r = r.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    return r

def make_sprocket(teeth, thickness, bore, pitch=12.7, chamfer=0):
    validate_positive(("teeth", teeth), ("thickness", thickness), ("bore", bore))
    pitch_dia = pitch * int(teeth) / math.pi
    od        = pitch_dia + pitch * 0.6
    if bore >= od:
        raise ValueError(f"Bore ({bore}mm) must be smaller than sprocket OD ({od:.1f}mm)")
    r = cq.Workplane("XY").circle(od / 2).extrude(thickness)
    r = r.faces(">Z").workplane().circle(bore / 2).cutThruAll()
    # Tooth cutouts approximation
    tooth_r = pitch / 2
    for i in range(int(teeth)):
        angle = math.radians(360 * i / int(teeth))
        cx = (pitch_dia / 2) * math.cos(angle)
        cy = (pitch_dia / 2) * math.sin(angle)
        try:
            r = r.faces(">Z").workplane().center(cx, cy).circle(tooth_r).cutThruAll()
        except Exception:
            pass
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_ball_bearing(id_, od, width):
    validate_positive(("id_", id_), ("od", od), ("width", width))
    validate_hollow(od, id_, "Ball Bearing")
    # Outer ring
    outer = cq.Workplane("XY").circle(od / 2).extrude(width)
    # Inner bore
    outer = outer.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    # Race groove (visual approximation)
    mid_r = (id_ + od) / 4
    return outer

def make_roller_bearing(id_, od, width):
    return make_ball_bearing(id_, od, width)

def make_sleeve_bearing(id_, od, length):
    validate_positive(("id_", id_), ("od", od), ("length", length))
    validate_hollow(od, id_, "Sleeve Bearing")
    r = cq.Workplane("XY").circle(od / 2).extrude(length)
    r = r.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    return r

def make_cuboid(length, width, height, chamfer=0, fillet=0):
    validate_positive(("length", length), ("width", width), ("height", height))
    r = cq.Workplane("XY").box(length, width, height)
    if chamfer > 0:
        try:
            r = r.edges().chamfer(chamfer)
        except Exception:
            pass
    elif fillet > 0:
        try:
            r = r.edges().fillet(fillet)
        except Exception:
            pass
    return r

def make_cylinder(diameter, height, chamfer=0):
    validate_positive(("diameter", diameter), ("height", height))
    r = cq.Workplane("XY").circle(diameter / 2).extrude(height)
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_hollow_cylinder(od, id_, length, chamfer=0):
    validate_positive(("od", od), ("id_", id_), ("length", length))
    validate_hollow(od, id_, "Hollow Cylinder")
    r = cq.Workplane("XY").circle(od / 2).extrude(length)
    r = r.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_flange(od, id_, thickness, chamfer=0):
    validate_positive(("od", od), ("id_", id_), ("thickness", thickness))
    validate_hollow(od, id_, "Flange")
    r = cq.Workplane("XY").circle(od / 2).extrude(thickness)
    r = r.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_flange_with_holes(od, id_, thickness, holes, hole_dia, pcd, chamfer=0):
    validate_positive(("od", od), ("id_", id_), ("thickness", thickness),
                      ("holes", holes), ("hole_dia", hole_dia), ("pcd", pcd))
    validate_hollow(od, id_, "Flange")
    validate_flange_holes(od, pcd, hole_dia)
    r = cq.Workplane("XY").circle(od / 2).extrude(thickness)
    r = r.faces(">Z").workplane().circle(id_ / 2).cutThruAll()
    r = (r.faces(">Z").workplane()
          .polarArray(pcd / 2, 0, 360, int(holes))
          .circle(hole_dia / 2).cutThruAll())
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_boss_mount(base_length, base_width, hole_dia, height, chamfer=0):
    validate_positive(("base_length", base_length), ("base_width", base_width),
                      ("hole_dia", hole_dia), ("height", height))
    base = cq.Workplane("XY").box(base_length, base_width, height * 0.3)
    boss_r = hole_dia * 1.5
    boss   = (cq.Workplane("XY")
               .workplane(offset=height * 0.3)
               .circle(boss_r).extrude(height * 0.7))
    r = base.union(boss)
    r = r.faces(">Z").workplane().circle(hole_dia / 2).cutThruAll()
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_connecting_rod(center_dist, big_end_dia, small_end_dia, thickness):
    validate_positive(("center_dist", center_dist), ("big_end_dia", big_end_dia),
                      ("small_end_dia", small_end_dia), ("thickness", thickness))
    big_r   = big_end_dia / 2
    small_r = small_end_dia / 2
    big_end   = cq.Workplane("XY").circle(big_r).extrude(thickness)
    small_end = cq.Workplane("XY").center(center_dist, 0).circle(small_r).extrude(thickness)
    rib_len = center_dist - big_r - small_r
    rib     = (cq.Workplane("XY")
                .center(big_r + rib_len / 2, 0)
                .box(rib_len, thickness * 0.5, thickness, centered=(True, True, False)))
    big_bore_r   = big_r * 0.5
    small_bore_r = small_r * 0.5
    big_end   = big_end.faces(">Z").workplane().circle(big_bore_r).cutThruAll()
    small_end = small_end.faces(">Z").workplane().circle(small_bore_r).cutThruAll()
    return big_end.union(small_end).union(rib)

def make_spring(wire_dia, coil_dia, turns, free_length):
    validate_positive(("wire_dia", wire_dia), ("coil_dia", coil_dia),
                      ("turns", turns), ("free_length", free_length))
    pitch  = free_length / max(turns, 1)
    radius = coil_dia / 2
    # Build as helix path + sweep circle
    import cadquery as cq2
    path   = (cq2.Workplane("XZ")
               .parametricCurve(
                   lambda t: (
                       radius * math.cos(2 * math.pi * turns * t),
                       radius * math.sin(2 * math.pi * turns * t),
                       free_length * t
                   ),
                   N=200
               ))
    wire_circle = cq2.Workplane("YZ").circle(wire_dia / 2)
    try:
        return wire_circle.sweep(path)
    except Exception:
        # fallback: cylinder approximation
        return cq.Workplane("XY").circle(coil_dia / 2).extrude(free_length)

def make_gear_hub(od, bore, thickness, teeth=0, chamfer=0):
    validate_positive(("od", od), ("bore", bore), ("thickness", thickness))
    if bore >= od:
        raise ValueError(f"Bore ({bore}mm) must be smaller than OD ({od}mm)")
    if teeth > 0:
        gear = SpurGear(module=od / max(teeth, 1), teeth_number=int(teeth), width=thickness, bore_d=bore)
        return gear.build()
    r = cq.Workplane("XY").circle(od / 2).extrude(thickness)
    r = r.faces(">Z").workplane().circle(bore / 2).cutThruAll()
    if chamfer > 0:
        try:
            r = r.faces(">Z").chamfer(chamfer)
        except Exception:
            pass
    return r

def make_cam(base_dia, lift, width):
    validate_positive(("base_dia", base_dia), ("lift", lift), ("width", width))
    base_r  = base_dia / 2
    total_r = base_r + lift
    # Eccentric circle approximation
    r = (cq.Workplane("XY")
          .ellipse(total_r, base_r)
          .extrude(width))
    return r

def make_flywheel(od, bore, thickness, chamfer=0):
    validate_positive(("od", od), ("bore", bore), ("thickness", thickness))
    if bore >= od:
        raise ValueError(f"Bore ({bore}mm) must be smaller than OD ({od}mm)")
    r = cq.Workplane("XY").circle(od / 2).extrude(thickness)
    r = r.faces(">Z").workplane().circle(bore / 2).cutThruAll()
    # Simple web pocket
    web_r = (od / 2) * 0.7
    web_h = thickness * 0.6
    try:
        pocket = (cq.Workplane("XY")
                   .workplane(offset=(thickness - web_h) / 2)
                   .circle(web_r).circle(bore * 0.8).extrude(web_h, combine=False))
        r = r.cut(pocket)
    except Exception:
        pass
    if chamfer > 0:
        try:
            r = r.edges("|Z").chamfer(chamfer)
        except Exception:
            pass
    return r

# ================== KEYWORD CLASSIFIER ==================
PART_KEYWORDS = {
    "shaft":            "shaft",
    "drive shaft":      "shaft",
    "spindle":          "shaft",
    "axle":             "axle",
    "pulley":           "pulley",
    "sheave":           "pulley",
    "coupling":         "coupling",
    "coupler":          "coupling",
    "nut":              "nut",
    "hex nut":          "nut",
    "bolt":             "bolt",
    "screw":            "bolt",
    "washer":           "washer",
    "dowel":            "pin",
    "taper pin":        "pin",
    "roll pin":         "pin",
    "pin":              "pin",
    "spur gear":        "spur_gear",
    "spur-gear":        "spur_gear",
    "bevel gear":       "bevel_gear",
    "ring gear":        "ring_gear",
    "annular gear":     "ring_gear",
    "helical gear":     "helical_gear",
    "worm gear":        "worm_gear",
    "worm":             "worm_gear",
    "clutch":           "clutch",
    "sprocket":         "sprocket",
    "chain gear":       "sprocket",
    "ball bearing":     "ball_bearing",
    "roller bearing":   "roller_bearing",
    "sleeve bearing":   "sleeve_bearing",
    "bush":             "sleeve_bearing",
    "bushing":          "sleeve_bearing",
    "cuboid":           "cuboid",
    "block":            "cuboid",
    "rectangular":      "cuboid",
    "cylinder":         "cylinder",
    "solid cylinder":   "cylinder",
    "hollow cylinder":  "hollow_cylinder",
    "pipe":             "hollow_cylinder",
    "tube":             "hollow_cylinder",
    "flange with holes":"flange_with_holes",
    "bolt flange":      "flange_with_holes",
    "flange":           "flange",
    "boss":             "boss_mount",
    "mount":            "boss_mount",
    "connecting rod":   "connecting_rod",
    "conrod":           "connecting_rod",
    "con rod":          "connecting_rod",
    "spring":           "spring",
    "coil spring":      "spring",
    "gear hub":         "gear_hub",
    "hub":              "gear_hub",
    "cam":              "cam",
    "flywheel":         "flywheel",
    "fly wheel":        "flywheel",
    # Parametric UI labels map to same keys
    "Shaft":            "shaft",
    "Pulley":           "pulley",
    "Coupling":         "coupling",
    "Axle":             "axle",
    "Nut":              "nut",
    "Bolt":             "bolt",
    "Washer":           "washer",
    "Pin":              "pin",
    "Spur Gear":        "spur_gear",
    "Bevel Gear":       "bevel_gear",
    "Ring Gear":        "ring_gear",
    "Helical Gear":     "helical_gear",
    "Worm Gear":        "worm_gear",
    "Clutch":           "clutch",
    "Sprocket":         "sprocket",
    "Ball Bearing":     "ball_bearing",
    "Roller Bearing":   "roller_bearing",
    "Sleeve Bearing":   "sleeve_bearing",
    "Cuboid":           "cuboid",
    "Cylinder":         "cylinder",
    "Hollow Cylinder":  "hollow_cylinder",
    "Flange":           "flange",
    "Flange with Holes":"flange_with_holes",
    "Boss / Mount":     "boss_mount",
    "Connecting Rod":   "connecting_rod",
    "Spring":           "spring",
    "Gear Hub":         "gear_hub",
    "Cam":              "cam",
    "Flywheel":         "flywheel",
}

def classify_part(prompt: str):
    p = prompt.lower()
    # Longer keywords first to avoid partial matches
    for kw in sorted(PART_KEYWORDS.keys(), key=len, reverse=True):
        if kw.lower() in p:
            return PART_KEYWORDS[kw]
    return None

# ================== DIMENSION EXTRACTOR ==================
PATTERNS = {
    "od":          r"(?:outer\s*diameter|od|outer\s*dia)\s*[:\-=]?\s*(\d+\.?\d*)",
    "id":          r"(?:inner\s*diameter|id|inner\s*dia|bore)\s*[:\-=]?\s*(\d+\.?\d*)",
    "diameter":    r"(?:diameter|dia)\s*[:\-=]?\s*(\d+\.?\d*)",
    "length":      r"(?:length|long|centre.to.centre\s*distance|center.to.center)\s*[:\-=]?\s*(\d+\.?\d*)",
    "width":       r"(?:width|wide|face\s*width)\s*[:\-=]?\s*(\d+\.?\d*)",
    "height":      r"(?:height|tall)\s*[:\-=]?\s*(\d+\.?\d*)",
    "thickness":   r"(?:thickness|thick|plate\s*thickness|rim\s*thickness)\s*[:\-=]?\s*(\d+\.?\d*)",
    "radius":      r"(?:radius|r)\s*[:\-=]?\s*(\d+\.?\d*)",
    "bore":        r"(\d+\.?\d*)\s*mm\s*bore|(?:bore\s*diameter|bore\s*dia)\s*[:\-=]?\s*(\d+\.?\d*)",
    "pcd":         r"(?:pitch\s*circle\s*diameter|pcd|bcd)\s*[:\-=]?\s*(\d+\.?\d*)",
    "holes":       r"(\d+)\s*(?:bolt\s*holes|holes)",
    "hole_dia":    r"(?:hole\s*diameter|hole\s*dia)\s*[:\-=]?\s*(\d+\.?\d*)",
    "teeth":       r"(\d+)\s*teeth",
    "module":      r"module\s*[:\-=]?\s*(\d+\.?\d*)",
    "face_width":  r"face\s*width\s*[:\-=]?\s*(\d+\.?\d*)",
    "helix_angle": r"helix\s*angle\s*[:\-=]?\s*(\d+\.?\d*)",
    "pitch_angle": r"pitch\s*angle\s*[:\-=]?\s*(\d+\.?\d*)",
    "lead_angle":  r"(?:lead\s*angle|worm\s*lead\s*angle)\s*[:\-=]?\s*(\d+\.?\d*)",
    "across_flats":r"(?:across\s*flats|af|width\s*across\s*flats)\s*[:\-=]?\s*(\d+\.?\d*)",
    "wire_dia":    r"wire\s*diameter\s*[:\-=]?\s*(\d+\.?\d*)",
    "coil_dia":    r"coil\s*diameter\s*[:\-=]?\s*(\d+\.?\d*)",
    "turns":       r"(?:number\s*of\s*active\s*turns|turns|active\s*turns)\s*[:\-=]?\s*(\d+\.?\d*)",
    "free_length": r"free\s*length\s*[:\-=]?\s*(\d+\.?\d*)",
    "grooves":     r"(?:number\s*of\s*grooves|grooves)\s*[:\-=]?\s*(\d+\.?\d*)",
    "plates":      r"(?:number\s*of\s*plates|plates)\s*[:\-=]?\s*(\d+\.?\d*)",
    "base_length": r"base\s*length\s*[:\-=]?\s*(\d+\.?\d*)",
    "base_width":  r"base\s*width\s*[:\-=]?\s*(\d+\.?\d*)",
    "lift":        r"(?:maximum\s*lift|lift)\s*[:\-=]?\s*(\d+\.?\d*)",
    "base_dia":    r"base\s*circle\s*diameter\s*[:\-=]?\s*(\d+\.?\d*)",
    "bore_a":      r"shaft\s*bore\s*a\s*[:\-=]?\s*(\d+\.?\d*)",
    "bore_b":      r"shaft\s*bore\s*b\s*[:\-=]?\s*(\d+\.?\d*)",
    "shoulder_dia":r"shoulder\s*diameter\s*[:\-=]?\s*(\d+\.?\d*)",
    "plate_thick": r"plate\s*thickness\s*[:\-=]?\s*(\d+\.?\d*)",
    "chamfer":     r"chamfer\s*[:\-=]?\s*(\d+\.?\d*)",
    "fillet":      r"fillet(?:\s*radius)?\s*[:\-=]?\s*(\d+\.?\d*)",
    "big_end_dia": r"big\s*end\s*diameter\s*[:\-=]?\s*(\d+\.?\d*)",
    "small_end_dia":r"small\s*end\s*diameter\s*[:\-=]?\s*(\d+\.?\d*)",
}

def extract_dimensions(prompt: str) -> dict:
    dims = {}
    p = prompt.lower()
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, p)
        if match:
            try:
                g = next(g for g in match.groups() if g is not None)
                dims[key] = int(float(g)) if key in ("teeth", "holes", "grooves", "plates", "turns") else float(g)
            except (StopIteration, ValueError):
                pass
    # radius → diameter conversion
    if "radius" in dims and "diameter" not in dims:
        dims["diameter"] = dims["radius"] * 2
    return dims

# ================== DEFAULT PARAMS ==================
DEFAULTS = {
    "shaft":            {"length": 200, "diameter": 25,  "chamfer": 0, "fillet": 0},
    "pulley":           {"od": 120, "bore": 25, "width": 40, "grooves": 1, "chamfer": 0},
    "coupling":         {"bore_a": 20, "bore_b": 20, "length": 80, "od": 60, "chamfer": 0},
    "axle":             {"length": 300, "diameter": 30, "shoulder_dia": 0, "chamfer": 0},
    "nut":              {"across_flats": 13, "height": 8, "hole_dia": 8, "chamfer": 0},
    "bolt":             {"diameter": 8, "length": 40, "head_height": 6, "across_flats": 13, "chamfer": 0},
    "washer":           {"od": 20, "id": 8.5, "thickness": 1.5, "chamfer": 0},
    "pin":              {"diameter": 8, "length": 40, "chamfer": 0},
    "spur_gear":        {"module": 2, "teeth": 20, "face_width": 30, "bore": 0, "chamfer": 0},
    "bevel_gear":       {"module": 2, "teeth": 20, "face_width": 20, "bore": 0, "pitch_angle": 45},
    "ring_gear":        {"module": 2, "teeth": 40, "face_width": 25},
    "helical_gear":     {"module": 2, "teeth": 24, "face_width": 35, "bore": 0, "helix_angle": 20},
    "worm_gear":        {"module": 3, "teeth": 40, "face_width": 30, "bore": 0, "lead_angle": 10},
    "clutch":           {"od": 150, "id": 50, "plate_thick": 3, "plates": 3},
    "sprocket":         {"teeth": 18, "thickness": 10, "bore": 20, "chamfer": 0},
    "ball_bearing":     {"id": 20, "od": 47, "width": 14},
    "roller_bearing":   {"id": 25, "od": 52, "width": 15},
    "sleeve_bearing":   {"id": 25, "od": 35, "length": 40},
    "cuboid":           {"length": 100, "width": 50, "height": 25, "chamfer": 0, "fillet": 0},
    "cylinder":         {"diameter": 50, "height": 80, "chamfer": 0},
    "hollow_cylinder":  {"od": 60, "id": 40, "length": 80, "chamfer": 0},
    "flange":           {"od": 100, "id": 30, "thickness": 15, "chamfer": 0},
    "flange_with_holes":{"od": 120, "id": 40, "thickness": 18, "holes": 6, "hole_dia": 10, "pcd": 90, "chamfer": 0},
    "boss_mount":       {"base_length": 80, "base_width": 60, "hole_dia": 20, "height": 30, "chamfer": 0},
    "connecting_rod":   {"center_dist": 200, "big_end_dia": 50, "small_end_dia": 30, "thickness": 12},
    "spring":           {"wire_dia": 2.5, "coil_dia": 25, "turns": 8, "free_length": 80},
    "gear_hub":         {"od": 60, "bore": 20, "thickness": 30, "teeth": 0, "chamfer": 0},
    "cam":              {"base_dia": 60, "lift": 15, "width": 20},
    "flywheel":         {"od": 300, "bore": 40, "thickness": 40, "chamfer": 0},
}

def fill_defaults(part: str, extracted: dict) -> dict:
    final = DEFAULTS.get(part, {}).copy()
    # Map frontend field names to backend field names
    alias = {
        "face_width": "face_width",
        "faceWidth":  "face_width",
        "id":         "id",
        "od":         "od",
        "bore":       "bore",
        "holeDia":    "hole_dia",
        "baseLength": "base_length",
        "baseWidth":  "base_width",
        "centerDist": "center_dist",
        "bigEndDia":  "big_end_dia",
        "smallEndDia":"small_end_dia",
        "wireWia":    "wire_dia",
        "coilDia":    "coil_dia",
        "freeLength": "free_length",
        "plateThick": "plate_thick",
        "acrossFlats":"across_flats",
        "boreA":      "bore_a",
        "boreB":      "bore_b",
        "shoulderDia":"shoulder_dia",
        "helixAngle": "helix_angle",
        "pitchAngle": "pitch_angle",
        "leadAngle":  "lead_angle",
        "baseDia":    "base_dia",
    }
    for fe_key, be_key in alias.items():
        if fe_key in extracted:
            extracted[be_key] = extracted.pop(fe_key)
    final.update({k: v for k, v in extracted.items() if v is not None})
    enforce_limits(final)
    return final

# ================== DISPATCHER ==================
def generate_from_template(prompt: str):
    part = classify_part(prompt)
    if not part:
        return None, "No template match"

    dims = extract_dimensions(prompt)
    p    = fill_defaults(part, dims)

    try:
        if   part == "shaft":            r = make_shaft(p["length"], p["diameter"], p.get("chamfer", 0), p.get("fillet", 0))
        elif part == "pulley":           r = make_pulley(p["od"], p["bore"], p["width"], p.get("grooves", 1), p.get("chamfer", 0))
        elif part == "coupling":         r = make_coupling(p["bore_a"], p["bore_b"], p["length"], p["od"], p.get("chamfer", 0))
        elif part == "axle":             r = make_axle(p["length"], p["diameter"], p.get("shoulder_dia", 0), p.get("chamfer", 0))
        elif part == "nut":              r = make_nut(p["across_flats"], p["height"], p["hole_dia"], p.get("chamfer", 0))
        elif part == "bolt":             r = make_bolt(p.get("diameter", 8), p["length"], p.get("head_height", 0), p.get("across_flats", 0), p.get("chamfer", 0))
        elif part == "washer":           r = make_washer(p["od"], p["id"], p["thickness"], p.get("chamfer", 0))
        elif part == "pin":              r = make_pin(p["diameter"], p["length"], p.get("chamfer", 0))
        elif part == "spur_gear":        r = make_spur_gear(p["module"], p["teeth"], p["face_width"], p.get("bore", 0), p.get("chamfer", 0))
        elif part == "bevel_gear":       r = make_bevel_gear(p["module"], p["teeth"], p["face_width"], p.get("bore", 0), p.get("pitch_angle", 45))
        elif part == "ring_gear":        r = make_ring_gear(p["module"], p["teeth"], p["face_width"])
        elif part == "helical_gear":     r = make_helical_gear(p["module"], p["teeth"], p["face_width"], p.get("bore", 0), p.get("helix_angle", 20))
        elif part == "worm_gear":        r = make_worm_gear(p["module"], p["teeth"], p["face_width"], p.get("bore", 0))
        elif part == "clutch":           r = make_clutch(p["od"], p["id"], p["plate_thick"], p.get("plates", 3))
        elif part == "sprocket":         r = make_sprocket(p["teeth"], p["thickness"], p["bore"], chamfer=p.get("chamfer", 0))
        elif part == "ball_bearing":     r = make_ball_bearing(p["id"], p["od"], p["width"])
        elif part == "roller_bearing":   r = make_roller_bearing(p["id"], p["od"], p["width"])
        elif part == "sleeve_bearing":   r = make_sleeve_bearing(p["id"], p["od"], p["length"])
        elif part == "cuboid":           r = make_cuboid(p["length"], p["width"], p["height"], p.get("chamfer", 0), p.get("fillet", 0))
        elif part == "cylinder":         r = make_cylinder(p["diameter"], p["height"], p.get("chamfer", 0))
        elif part == "hollow_cylinder":  r = make_hollow_cylinder(p["od"], p["id"], p["length"], p.get("chamfer", 0))
        elif part == "flange":           r = make_flange(p["od"], p["id"], p["thickness"], p.get("chamfer", 0))
        elif part == "flange_with_holes":r = make_flange_with_holes(p["od"], p["id"], p["thickness"], p["holes"], p["hole_dia"], p["pcd"], p.get("chamfer", 0))
        elif part == "boss_mount":       r = make_boss_mount(p["base_length"], p["base_width"], p["hole_dia"], p["height"], p.get("chamfer", 0))
        elif part == "connecting_rod":   r = make_connecting_rod(p["center_dist"], p["big_end_dia"], p["small_end_dia"], p["thickness"])
        elif part == "spring":           r = make_spring(p["wire_dia"], p["coil_dia"], p["turns"], p["free_length"])
        elif part == "gear_hub":         r = make_gear_hub(p["od"], p["bore"], p["thickness"], p.get("teeth", 0), p.get("chamfer", 0))
        elif part == "cam":              r = make_cam(p["base_dia"], p["lift"], p["width"])
        elif part == "flywheel":         r = make_flywheel(p["od"], p["bore"], p["thickness"], p.get("chamfer", 0))
        else:
            return None, f"Dispatcher: no handler for '{part}'"
        return r, None
    except Exception as e:
        return None, str(e)

# ================== AI FALLBACK ==================
def generate_with_ai(prompt: str, error: str = None) -> str:
    error_ctx = f"\nPrevious error: {error}\nFix it in your new code." if error else ""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"""
You are a CadQuery expert. Write Python code using CadQuery to build this mechanical part:
{prompt}
{error_ctx}

Rules:
- Import cadquery at top: import cadquery as cq
- Store final shape in variable called 'result'
- Do NOT use any file I/O, show(), input(), or external imports
- Available: cadquery as cq, math
- Available gear library: from cq_gears import SpurGear, RingGear, BevelGear
- Return ONLY Python code — no markdown, no backticks, no explanation

Example for a spur gear:
from cq_gears import SpurGear
gear = SpurGear(module=2, teeth_number=20, width=30)
result = gear.build()
"""}]
    )
    return response.choices[0].message.content.strip()

def run_ai_with_retry(prompt: str):
    error = None
    for attempt in range(3):
        code = generate_with_ai(prompt, error)
        # Strip markdown fences if present
        code = re.sub(r"^```(?:python)?\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
        try:
            safe, pattern = is_code_safe(code)
            if not safe:
                return None, f"Blocked unsafe code pattern: {pattern}"
            exec_locals = {}
            exec(code, SAFE_GLOBALS, exec_locals)
            if "result" not in exec_locals:
                raise ValueError("AI code did not produce a 'result' variable")
            return exec_locals["result"], None
        except Exception as e:
            error = str(e)
            print(f"AI attempt {attempt + 1} failed: {error}")
    return None, f"AI failed after 3 attempts. Last error: {error}"

# ================== ROUTE ==================
@app.route("/generate", methods=["POST"])
def generate():
    data      = request.json or {}
    prompt    = data.get("prompt", "").strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not prompt:
        return jsonify({"status": "failed", "error": "No prompt provided"}), 400

    result    = None
    error_msg = None
    method    = None

    # 1. Try template
    result, err = generate_from_template(prompt)
    if result:
        method = "template"
    else:
        print(f"Template failed ({err}), falling back to AI...")
        # 2. AI fallback
        result, err = run_ai_with_retry(prompt)
        if result:
            method = "ai"
        else:
            error_msg = err

    if not result:
        return jsonify({
            "status":    "failed",
            "method":    None,
            "error":     error_msg or "Generation failed",
            "step_b64":  None,
            "stl_b64":   None,
            "timestamp": timestamp,
        }), 500

    try:
        os.makedirs("outputs", exist_ok=True)
        step_path = f"outputs/part_{timestamp}.step"
        stl_path  = f"outputs/part_{timestamp}.stl"
        cq.exporters.export(result, step_path)
        cq.exporters.export(result, stl_path)

        with open(step_path, "rb") as f:
            step_b64 = base64.b64encode(f.read()).decode()
        with open(stl_path, "rb") as f:
            stl_b64  = base64.b64encode(f.read()).decode()

        return jsonify({
            "status":    "success",
            "method":    method,
            "error":     None,
            "step_b64":  step_b64,
            "stl_b64":   stl_b64,
            "timestamp": timestamp,
        })
    except Exception as e:
        return jsonify({
            "status":    "failed",
            "method":    method,
            "error":     f"Export failed: {str(e)}",
            "step_b64":  None,
            "stl_b64":   None,
            "timestamp": timestamp,
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
