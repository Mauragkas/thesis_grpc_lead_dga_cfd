use super::super::gen::{NodeAddr as ProtoNode, RangeEntry};
use crate::ring::NodeAddr;

pub(super) fn to_proto(n: &NodeAddr) -> ProtoNode {
    ProtoNode {
        id: n.id,
        address: n.address.clone(),
    }
}

pub(super) fn from_proto(n: ProtoNode) -> NodeAddr {
    NodeAddr {
        id: n.id,
        address: n.address,
    }
}

pub(super) fn entries_to_proto(v: &[(String, String)]) -> Vec<RangeEntry> {
    v.iter()
        .map(|(k, val)| RangeEntry {
            key: k.clone(),
            value: val.clone(),
        })
        .collect()
}

pub(super) fn entries_from_proto(v: Vec<RangeEntry>) -> Vec<(String, String)> {
    v.into_iter().map(|e| (e.key, e.value)).collect()
}
