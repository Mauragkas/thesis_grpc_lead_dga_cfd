use serde::{Deserialize, Serialize};

/// Active feature names matching the aerodynamic surrogate model in `test_gp.py`.
pub const ACTIVE_FEATURE_NAMES: [&str; 11] = [
    "wing_span",
    "wing_root_chord",
    "wing_tip_chord",
    "wing_sweep",
    "wing_dihedral",
    "wing_twist",
    "wing_x_pos",
    "fuse_length",
    "fuse_max_diam",
    "naca_m",
    "naca_t",
];

/// Parameter configuration representing geometric dimensions of the aircraft.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AircraftConfig {
    pub fuse_length: f64,
    pub fuse_max_diam: f64,
    pub nose_ratio: f64,
    pub tail_ratio: f64,
    pub wing_span: f64,
    pub wing_root_chord: f64,
    pub wing_tip_chord: f64,
    pub wing_sweep: f64,
    pub wing_dihedral: f64,
    pub wing_twist: f64,
    pub wing_x_pos: f64,
    pub wing_z_pos: f64,
    pub naca_m: f64,
    pub naca_p: f64,
    pub naca_t: f64,
    pub tail_x_pos: f64,
    pub v_stab_height: f64,
    pub v_stab_root: f64,
    pub v_stab_tip: f64,
    pub h_stab_span: f64,
    pub h_stab_root: f64,
    pub h_stab_tip: f64,
}

impl AircraftConfig {
    /// Extracts the 11 active features used for aerodynamic surrogate regression.
    pub fn to_active_features(&self) -> Vec<f64> {
        vec![
            self.wing_span,
            self.wing_root_chord,
            self.wing_tip_chord,
            self.wing_sweep,
            self.wing_dihedral,
            self.wing_twist,
            self.wing_x_pos,
            self.fuse_length,
            self.fuse_max_diam,
            self.naca_m,
            self.naca_t,
        ]
    }

    /// Extracts all 22 configuration parameters as a feature vector.
    pub fn to_all_features(&self) -> Vec<f64> {
        vec![
            self.fuse_length,
            self.fuse_max_diam,
            self.nose_ratio,
            self.tail_ratio,
            self.wing_span,
            self.wing_root_chord,
            self.wing_tip_chord,
            self.wing_sweep,
            self.wing_dihedral,
            self.wing_twist,
            self.wing_x_pos,
            self.wing_z_pos,
            self.naca_m,
            self.naca_p,
            self.naca_t,
            self.tail_x_pos,
            self.v_stab_height,
            self.v_stab_root,
            self.v_stab_tip,
            self.h_stab_span,
            self.h_stab_root,
            self.h_stab_tip,
        ]
    }
}
