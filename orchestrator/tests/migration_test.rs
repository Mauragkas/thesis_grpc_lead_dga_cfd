//! Integration tests for the migration module: migrant selection.

use orchestrator::migration::{MigrantSelector, TopKSelector};

#[test]
fn top_k_selects_highest_fitness() {
    let pop = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let fit = vec![10.0, 30.0, 20.0, 40.0];
    let migrants = TopKSelector.select(&pop, &fit, 2);
    assert_eq!(migrants.len(), 2);
    assert_eq!(migrants[0].genes, vec![3.0]); // fitness 40
    assert_eq!(migrants[1].genes, vec![1.0]); // fitness 30
}
