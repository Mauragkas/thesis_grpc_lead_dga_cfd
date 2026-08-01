/// SRP: a plain data record describing one evaluated individual.
/// No behaviour lives here; other modules operate on it.
#[derive(Debug, Clone)]
pub struct GeneRecord {
    pub genes: Vec<f64>,
    pub fitness: f64,
    /// GA generation at which the evaluation occurred.
    pub generation: usize,
    /// Last generation in which this record was retrieved by a query.
    /// Refreshed on every KNN hit (access-refresh TTL semantics).
    pub last_accessed_gen: usize,
}
