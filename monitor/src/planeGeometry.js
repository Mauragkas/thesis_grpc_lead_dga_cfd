/**
 * Plane geometry generation and STL serialization for loadbalanceGA.
 * Compatible with Node.js and browser environments.
 */

const GENE_BOUNDS = [
  { name: 'wing_span',       low: 80.0,  high: 220.0, unit: 'mm' },
  { name: 'wing_root_chord', low: 35.0,  high: 75.0,  unit: 'mm' },
  { name: 'wing_tip_chord',  low: 10.0,  high: 45.0,  unit: 'mm' },
  { name: 'wing_sweep',      low: 0.0,   high: 25.0,  unit: '°' },
  { name: 'wing_dihedral',   low: 0.0,   high: 10.0,  unit: '°' },
  { name: 'wing_twist',      low: -6.0,  high: 2.0,   unit: '°' },
  { name: 'wing_x_pos',      low: 55.0,  high: 95.0,  unit: 'mm' },
  { name: 'naca_m',          low: 0.0,   high: 5.0,   unit: '%' },
  { name: 'fuse_length',     low: 200.0, high: 350.0, unit: 'mm' },
  { name: 'fuse_max_diam',   low: 8.0,   high: 16.0,  unit: 'mm' },
];

const BASELINE = {
  fuse_length: 250.0,
  fuse_max_diam: 10.0,
  nose_ratio: 0.25,
  tail_ratio: 0.35,
  wing_span: 140.0,
  wing_root_chord: 55.0,
  wing_tip_chord: 25.0,
  wing_sweep: 12.0,
  wing_dihedral: 4.0,
  wing_twist: -3.0,
  wing_x_pos: 75.0,
  wing_z_pos: -2.0,
  naca_m: 2.0,
  naca_p: 4.0,
  naca_t: 12.0,
  tail_x_pos: 210.0,
  v_stab_height: 45.0,
  v_stab_root: 35.0,
  v_stab_tip: 18.0,
  h_stab_span: 45.0,
  h_stab_root: 28.0,
  h_stab_tip: 15.0,
};

function decodeGenes(genes) {
  const params = { ...BASELINE };
  if (!Array.isArray(genes)) return params;
  for (let i = 0; i < GENE_BOUNDS.length && i < genes.length; i++) {
    const bound = GENE_BOUNDS[i];
    const u = Math.max(0.0, Math.min(1.0, Number(genes[i]) || 0.0));
    params[bound.name] = bound.low + (bound.high - bound.low) * u;
  }
  return params;
}

function deg2rad(deg) {
  return (deg * Math.PI) / 180.0;
}

function nacaYt(x, t) {
  return (
    5 *
    t *
    (0.2969 * Math.sqrt(Math.max(0, x)) -
      0.126 * x -
      0.3516 * x * x +
      0.2843 * x * x * x -
      0.1015 * x * x * x * x)
  );
}

function nacaYc(x, m, p) {
  if (p === 0 || m === 0) return 0;
  if (x < p) {
    return (m / (p * p)) * (2 * p * x - x * x);
  }
  return (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * x - x * x);
}

function nacaDycDx(x, m, p) {
  if (p === 0 || m === 0) return 0;
  if (x < p) {
    return ((2 * m) / (p * p)) * (p - x);
  }
  return ((2 * m) / ((1 - p) * (1 - p))) * (p - x);
}

function naca4Points(mNum, pNum, tNum, n = 24) {
  const m = mNum / 100.0;
  const p = pNum / 10.0;
  const t = tNum / 100.0;

  const upper = [];
  const lower = [];

  for (let i = 0; i <= n; i++) {
    const beta = (i * Math.PI) / n;
    const x = 0.5 * (1 - Math.cos(beta));
    const yt = nacaYt(x, t);
    const yc = nacaYc(x, m, p);
    const dyc = nacaDycDx(x, m, p);
    const theta = Math.atan(dyc);

    upper.push([x - yt * Math.sin(theta), yc + yt * Math.cos(theta)]);
    lower.push([x + yt * Math.sin(theta), yc - yt * Math.cos(theta)]);
  }

  const points = [];
  for (let i = n; i >= 0; i--) {
    points.push(upper[i]);
  }
  for (let i = 1; i <= n; i++) {
    points.push(lower[i]);
  }
  return points;
}

class MeshBuilder {
  constructor() {
    this.positions = []; // flat [x, y, z, ...]
    this.indices = [];   // flat [i, j, k, ...]
  }

  addVertex(x, y, z) {
    const idx = this.positions.length / 3;
    this.positions.push(x, y, z);
    return idx;
  }

  addTriangle(a, b, c) {
    this.indices.push(a, b, c);
  }

  addQuad(a, b, c, d) {
    this.indices.push(a, b, c);
    this.indices.push(a, c, d);
  }
}

/**
 * Builds full 3D geometry of the parametric aircraft.
 */
function buildPlaneMesh(params) {
  const builder = new MeshBuilder();

  buildFuselage(builder, params);
  buildWings(builder, params);
  buildTail(builder, params);

  return {
    positions: new Float32Array(builder.positions),
    indices: new Uint32Array(builder.indices),
  };
}

function buildFuselage(b, params) {
  const L = params.fuse_length;
  const rMax = params.fuse_max_diam / 2.0;
  const lNose = L * params.nose_ratio;
  const lTail = L * params.tail_ratio;
  const lMid = L - lNose - lTail;

  const nRadial = 20;
  const nNoseSteps = 10;
  const nMidSteps = 6;
  const nTailSteps = 12;

  const stations = [];

  // 1. Nose cone: elliptic profile (i starts from 1; tipFront at (0,0,0) caps the front)
  for (let i = 1; i <= nNoseSteps; i++) {
    const u = i / nNoseSteps;
    const x = u * lNose;
    const r = rMax * Math.sqrt(Math.max(0, 1 - Math.pow(1 - u, 2)));
    stations.push({ x, r });
  }

  // 2. Mid body: constant cylinder
  for (let i = 1; i <= nMidSteps; i++) {
    const u = i / nMidSteps;
    const x = lNose + u * lMid;
    stations.push({ x, r: rMax });
  }

  // 3. Tail cone: linear taper to 15% diameter
  for (let i = 1; i <= nTailSteps; i++) {
    const u = i / nTailSteps;
    const x = lNose + lMid + u * lTail;
    const r = rMax * (1.0 - 0.85 * u);
    stations.push({ x, r });
  }

  // Create rings of vertices
  const ringIndices = [];
  for (const s of stations) {
    const ring = [];
    for (let j = 0; j < nRadial; j++) {
      const phi = (j * 2 * Math.PI) / nRadial;
      const y = s.r * Math.cos(phi);
      const z = s.r * Math.sin(phi);
      ring.push(b.addVertex(s.x, y, z));
    }
    ringIndices.push(ring);
  }

  // Connect adjacent rings
  for (let i = 0; i < ringIndices.length - 1; i++) {
    const r1 = ringIndices[i];
    const r2 = ringIndices[i + 1];
    for (let j = 0; j < nRadial; j++) {
      const jNext = (j + 1) % nRadial;
      b.addQuad(r1[j], r1[jNext], r2[jNext], r2[j]);
    }
  }

  // Front cap (nose tip)
  const tipFront = b.addVertex(0, 0, 0);
  const rFront = ringIndices[0];
  for (let j = 0; j < nRadial; j++) {
    const jNext = (j + 1) % nRadial;
    b.addTriangle(tipFront, rFront[j], rFront[jNext]);
  }

  // Rear cap (tail tip)
  const lastRing = ringIndices[ringIndices.length - 1];
  const lastX = stations[stations.length - 1].x;
  const tipBack = b.addVertex(lastX, 0, 0);
  for (let j = 0; j < nRadial; j++) {
    const jNext = (j + 1) % nRadial;
    b.addTriangle(tipBack, lastRing[jNext], lastRing[j]);
  }
}

function buildWings(b, params) {
  const span = params.wing_span;
  const rootC = params.wing_root_chord;
  const tipC = params.wing_tip_chord;
  const sweepRad = deg2rad(params.wing_sweep);
  const dihedralRad = deg2rad(params.wing_dihedral);
  const twistRad = deg2rad(params.wing_twist);
  const xPos = params.wing_x_pos;
  const zPos = params.wing_z_pos;

  const airfoil = naca4Points(params.naca_m, params.naca_p, params.naca_t, 20);
  const nStations = 12;

  // Starboard (Right, Y > 0)
  buildWingPanel(b, airfoil, nStations, 1.0, span, rootC, tipC, sweepRad, dihedralRad, twistRad, xPos, zPos);
  // Port (Left, Y < 0)
  buildWingPanel(b, airfoil, nStations, -1.0, span, rootC, tipC, sweepRad, dihedralRad, twistRad, xPos, zPos);
}

function buildWingPanel(b, airfoil, nStations, sideSign, span, rootC, tipC, sweep, dihedral, twist, xPos, zPos) {
  const stationRings = [];

  for (let i = 0; i <= nStations; i++) {
    const s = i / nStations;
    const chord = rootC * (1 - s) + tipC * s;
    const xLe = xPos + s * span * Math.tan(sweep);
    const yStation = sideSign * s * span;
    const zLe = zPos + s * span * Math.tan(dihedral);
    const t = s * twist; // washout angle

    const ring = [];
    for (const [px, py] of airfoil) {
      const rx = px * Math.cos(t) - py * Math.sin(t);
      const rz = px * Math.sin(t) + py * Math.cos(t);

      const vx = xLe + rx * chord;
      const vy = yStation;
      const vz = zLe + rz * chord;
      ring.push(b.addVertex(vx, vy, vz));
    }
    stationRings.push(ring);
  }

  // Loft skin between chord stations
  const nPts = airfoil.length;
  for (let i = 0; i < nStations; i++) {
    const r1 = stationRings[i];
    const r2 = stationRings[i + 1];
    for (let j = 0; j < nPts; j++) {
      const jNext = (j + 1) % nPts;
      if (sideSign > 0) {
        b.addQuad(r1[j], r2[j], r2[jNext], r1[jNext]);
      } else {
        b.addQuad(r1[j], r1[jNext], r2[jNext], r2[j]);
      }
    }
  }

  // Root & Tip caps (fan triangulation)
  const rootRing = stationRings[0];
  const tipRing = stationRings[nStations];
  fanCap(b, rootRing, sideSign > 0);
  fanCap(b, tipRing, sideSign < 0);
}

function fanCap(b, ring, flipNormal) {
  const n = ring.length;
  for (let i = 1; i < n - 1; i++) {
    if (flipNormal) {
      b.addTriangle(ring[0], ring[i + 1], ring[i]);
    } else {
      b.addTriangle(ring[0], ring[i], ring[i + 1]);
    }
  }
}

function buildTail(b, params) {
  const tailX = params.tail_x_pos;
  const airfoil = naca4Points(0, 0, 10, 16); // Symmetric NACA 0010

  // 1. Horizontal Stabilizer (Starboard & Port)
  const hSpan = params.h_stab_span;
  const hRoot = params.h_stab_root;
  const hTip = params.h_stab_tip;
  buildWingPanel(b, airfoil, 6, 1.0, hSpan, hRoot, hTip, 0, 0, 0, tailX, 0);
  buildWingPanel(b, airfoil, 6, -1.0, hSpan, hRoot, hTip, 0, 0, 0, tailX, 0);

  // 2. Vertical Stabilizer (Upward along +Z)
  buildVerticalFin(b, airfoil, params.v_stab_height, params.v_stab_root, params.v_stab_tip, tailX);
}

function buildVerticalFin(b, airfoil, height, rootC, tipC, xPos) {
  const nStations = 6;
  const stationRings = [];

  for (let i = 0; i <= nStations; i++) {
    const s = i / nStations;
    const chord = rootC * (1 - s) + tipC * s;
    const xLe = xPos + s * (rootC - tipC);
    const zStation = s * height;

    const ring = [];
    for (const [px, py] of airfoil) {
      const vx = xLe + px * chord;
      const vy = py * chord;
      const vz = zStation;
      ring.push(b.addVertex(vx, vy, vz));
    }
    stationRings.push(ring);
  }

  const nPts = airfoil.length;
  for (let i = 0; i < nStations; i++) {
    const r1 = stationRings[i];
    const r2 = stationRings[i + 1];
    for (let j = 0; j < nPts; j++) {
      const jNext = (j + 1) % nPts;
      b.addQuad(r1[j], r2[j], r2[jNext], r1[jNext]);
    }
  }

  fanCap(b, stationRings[0], true);
  fanCap(b, stationRings[nStations], false);
}

/**
 * Converts mesh data (positions & indices) into an ASCII STL string.
 */
function meshToSTL(mesh, solidName = 'aircraft') {
  const { positions, indices } = mesh;
  let out = `solid ${solidName}\n`;

  for (let i = 0; i < indices.length; i += 3) {
    const i1 = indices[i] * 3;
    const i2 = indices[i + 1] * 3;
    const i3 = indices[i + 2] * 3;

    const ax = positions[i1],     ay = positions[i1 + 1], az = positions[i1 + 2];
    const bx = positions[i2],     by = positions[i2 + 1], bz = positions[i2 + 2];
    const cx = positions[i3],     cy = positions[i3 + 1], cz = positions[i3 + 2];

    // Compute normal
    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-9) {
      nx /= len;
      ny /= len;
      nz /= len;
    } else {
      nx = 0; ny = 0; nz = 1;
    }

    out += `  facet normal ${nx.toFixed(6)} ${ny.toFixed(6)} ${nz.toFixed(6)}\n`;
    out += `    outer loop\n`;
    out += `      vertex ${ax.toFixed(4)} ${ay.toFixed(4)} ${az.toFixed(4)}\n`;
    out += `      vertex ${bx.toFixed(4)} ${by.toFixed(4)} ${bz.toFixed(4)}\n`;
    out += `      vertex ${cx.toFixed(4)} ${cy.toFixed(4)} ${cz.toFixed(4)}\n`;
    out += `    endloop\n`;
    out += `  endfacet\n`;
  }

  out += `endsolid ${solidName}\n`;
  return out;
}

// Universal export (Node.js CommonJS and ES/Browser)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    GENE_BOUNDS,
    BASELINE,
    decodeGenes,
    buildPlaneMesh,
    meshToSTL,
  };
}
if (typeof window !== 'undefined') {
  window.PlaneGeometry = {
    GENE_BOUNDS,
    BASELINE,
    decodeGenes,
    buildPlaneMesh,
    meshToSTL,
  };
}
