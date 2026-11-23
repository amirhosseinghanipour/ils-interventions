"""Check the exact reference graph and the effective candidate budgets after organizer pruning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from eap.graph import Graph

from scripts.mib_official.public_ioi import (
    GRID,
    MODES,
    apply,
    boundary_mask,
    rank_edges,
)


def mask_hash(graph):
    bits = graph.in_graph.numpy().tobytes() + graph.nodes_in_graph.numpy().tobytes()
    return hashlib.sha256(bits).hexdigest()


def audit(path):
    graph = Graph.from_json(str(path))
    output = {"legal_edges": len(graph.edges), "modes": {}}
    for mode in MODES:
        rank = rank_edges(graph, mode == "CMD")
        entries = {}
        for i, fraction in enumerate(GRID[:-1]):
            size = int(fraction * len(graph.edges))
            graph.apply_topn(size, absolute=mode == "CMD")
            edges, nodes = graph.in_graph.clone(), graph.nodes_in_graph.clone()
            measured = [0] + np.random.default_rng(92144 + i).choice(
                np.arange(1, 64), 23, replace=False
            ).tolist()
            hashes, retained_counts = [], []
            for code in range(64):
                chosen = boundary_mask(rank, size, code)
                assert len(chosen.unique()) == size and graph.real_edge_mask.flatten()[chosen].all()
                apply(graph, chosen)
                hashes.append(mask_hash(graph))
                retained_counts.append(graph.count_included_edges())
                if code == 0:
                    assert torch.equal(edges, graph.in_graph) and torch.equal(nodes, graph.nodes_in_graph)
            entries[str(fraction)] = {
                "nominal_edges": size,
                "reference_edges_after_pruning": int(edges.sum()),
                "reference_nodes_after_pruning": int(nodes.sum()),
                "reference_node_and_edge_masks_equal": True,
                "reference_sha256": hashes[0],
                "distinct_effective_candidates_of_64": len(set(hashes)),
                "distinct_effective_measured_of_24": len({hashes[j] for j in measured}),
                "candidate_edges_after_pruning_range": [min(retained_counts), max(retained_counts)],
            }
        output["modes"][mode] = entries
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("organizer_graph", type=Path)
    print(json.dumps(audit(p.parse_args().organizer_graph), indent=2))
