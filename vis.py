from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.container import Docker
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.network import Envoy

graph_attr = {
    "fontsize": "18",
    "fontname": "Helvetica-Bold",
    "fontcolor": "#0f172a",
    "bgcolor": "#ffffff",
    "pad": "0.8",
    "nodesep": "0.8",
    "ranksep": "1.2",
    "splines": "ortho",
}

node_attr = {
    "fontsize": "11",
    "fontname": "Helvetica-Bold",
    "fontcolor": "#0f172a",
}

edge_attr = {
    "fontsize": "9",
    "fontname": "Helvetica",
    "fontcolor": "#475569",
    "color": "#64748b",
    "penwidth": "1.5",
}

cluster_style = {
    "fontsize": "12",
    "fontname": "Helvetica-Bold",
    "fontcolor": "#0f172a",
    "bgcolor": "#f8fafc",
    "color": "#cbd5e1",
    "style": "rounded,filled",
    "penwidth": "1.5",
    "margin": "20",
}

with Diagram(
    "DGA Island Architecture & gRPC Evaluator Grid (Refactored)",
    show=False,
    filename="architecture_grpc_envoy_fixed",
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    with Cluster("Distributed Storage Layer", graph_attr=cluster_style):
        lead = Docker("LEAD Node Container\n(Learned DHT Index)")

    with Cluster("Island Control Layer", graph_attr=cluster_style):
        orchestrator = Docker("Orchestrator\n(TARGET=lb:50051)")
        cache = Redis("Local State Cache\n(Checkpointing & Fitness)")

    with Cluster("Elastic Evaluator Grid", graph_attr=cluster_style):
        lb = Envoy("Envoy Proxy\n(Ingress Listener :50051)")
        worker_pool = Docker("Evaluator Worker Pool\n(xN Autoscaled Pods\nContainer Port :50051)")

    orchestrator >> Edge(color="#2563eb", penwidth="2.0", label=" 1. DHT Lookup / Similarity Query") >> lead
    orchestrator >> Edge(style="dashed", label=" 2. Checkpoint Task State") >> cache
    orchestrator >> Edge(color="#059669", penwidth="2.0", label=" 3. gRPC Eval Request\n(Keepalive Enabled)") >> lb
    lb >> Edge(color="#059669", label=" 4. Least Request Dispatch\n(Container Network)") >> worker_pool
