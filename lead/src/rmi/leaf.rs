use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Anchor
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Anchor {
    pub offset: f64,
    pub scale: f64,
}

impl Default for Anchor {
    fn default() -> Self {
        Self {
            offset: 0.0,
            scale: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// LinearLeaf
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LinearLeaf {
    pub weight: f64,
    pub bias: f64,
    pub anchor: Anchor,
}

// ---------------------------------------------------------------------------
// RadixSpline — 32-way radix table for CDF prediction
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RadixSplineLeaf {
    pub radix_table: Vec<u32>, // 2^RP entries mapping prefix -> rank
    pub rp: usize,             // radix prefix bits (default 10 -> 1024 entries)
    pub anchor: Anchor,
}

pub const RP: usize = 10;
pub const RADIX_ENTRIES: usize = 1 << RP;

// ---------------------------------------------------------------------------
// LeafKind enum
// ---------------------------------------------------------------------------
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum LeafKind {
    Linear(LinearLeaf),
    RadixSpline(RadixSplineLeaf),
}

impl LeafKind {
    pub fn anchor(&self) -> &Anchor {
        match self {
            LeafKind::Linear(l) => &l.anchor,
            LeafKind::RadixSpline(r) => &r.anchor,
        }
    }

    pub fn anchor_mut(&mut self) -> &mut Anchor {
        match self {
            LeafKind::Linear(l) => &mut l.anchor,
            LeafKind::RadixSpline(r) => &mut r.anchor,
        }
    }

    pub fn predict(&self, feature: f64) -> f64 {
        let raw = match self {
            LeafKind::Linear(l) => l.weight * feature + l.bias,
            LeafKind::RadixSpline(r) => {
                let prefix = (feature * (RADIX_ENTRIES as f64)) as usize;
                let idx = prefix.min(RADIX_ENTRIES - 1);
                (r.radix_table[idx] as f64) / (u32::MAX as f64)
            }
        };
        let anchor = self.anchor();
        let y = raw * anchor.scale + anchor.offset;
        if !y.is_finite() {
            0.0
        } else {
            y.clamp(0.0, 1.0)
        }
    }
}
