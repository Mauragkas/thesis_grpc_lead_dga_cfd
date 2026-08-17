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
        self.learning.model_for_version(version).await
    }
}
