use tracing::info;

use crate::storage::KeyStore;
use crate::transport::RemoteNode;

use super::super::node::LeadNode;

impl<S, R> LeadNode<S, R>
where
    S: KeyStore,
    R: RemoteNode,
{
    pub async fn join(&self, known: &str) {
        let max_extra = self.config.successor_list_len.saturating_sub(1);
        for vnode in &self.vnodes {
            if let Some(succ) = self
                .remote
                .find_successor(known, vnode.vid, vnode.vid)
                .await
            {
                let mut list = vec![succ.clone()];
                let remote_list = self.remote.get_successor_list(&succ.address, succ.id).await;
                for n in remote_list.into_iter().take(max_extra) {
                    if n.id != succ.id && !list.contains(&n) {
                        list.push(n);
                    }
                }
                *vnode.successor_list.write().await = list;
                vnode.fingers.write().await[0] = Some(succ);
                vnode.mark_active().await;
            }
        }
        info!("joined ring with {} vnodes", self.vnodes.len());
    }
}
