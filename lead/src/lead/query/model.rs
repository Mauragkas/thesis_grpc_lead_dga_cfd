use crate::rmi::RmiModel;
use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::node::LeadNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub(super) async fn get_model_for_version(&self, version: u64) -> RmiModel {
        let rmi = self.rmi.read().await;
        if rmi.active.version == version {
            rmi.active.clone()
        } else if let Some(ref update) = rmi.update {
            if update.version == version {
                update.clone()
            } else {
                rmi.active.clone() // fallback
            }
        } else {
            rmi.active.clone()
        }
    }
}
