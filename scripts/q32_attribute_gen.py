"""Generate Q32 CIM attribute JSON for subgraph-to-core mapping.

Assigns GEMM (systolic array) subgraphs to core 0 (Q32 tensor core)
and vector subgraphs to core 1 (DSP / Vision 341).

Usage:
    python scripts/q32_attribute_gen.py <onnx_path> <output_path>
"""
import json
import os
import sys

try:
    import onnx
except ImportError:
    onnx = None


def generate_q32_attributes(onnx_path, output_path):
    """Assign GEMM subgraphs to core 0, vector subgraphs to core 1.

    Subgraph IDs are auto-assigned by TileGraphParser based on parallel
    loop iteration order.  Without full C++ parsing, we use a heuristic:
    scan compute nodes for their compute_type attribute and build a
    per-subgraph mapping.

    For pure GEMM workloads (matmul), all subgraphs go to core 0.
    For mixed workloads, the outermost parallel loop determines subgraph
    boundaries; each iteration is one subgraph.
    """
    if onnx is None:
        # Fallback: assign all subgraphs to core 0
        _write_default_attribute(output_path, num_subgraphs=1)
        return

    model = onnx.load(onnx_path)
    graph = model.graph

    # Collect compute_type per subgraph.
    # Subgraphs are delineated by outer parallel loops (loop_type == 1).
    # We walk the ONNX nodes in order and track parallel loop boundaries.
    subgraph_compute_types = {}
    current_subgraph = 0
    parallel_loop_depth = 0

    for node in graph.node:
        attrs = {a.name: a for a in node.attribute}

        if node.op_type == "loop_index_node":
            loop_type = attrs.get("torchsim_loop_type")
            if loop_type is not None and loop_type.i == 1:  # PARALLEL_LOOP
                parallel_loop_depth += 1

        elif node.op_type == "loop_end_node":
            if parallel_loop_depth > 0:
                parallel_loop_depth -= 1
                if parallel_loop_depth == 0:
                    current_subgraph += 1

        elif node.op_type == "compute_node":
            ct_attr = attrs.get("torchsim_compute_type")
            compute_type = ct_attr.i if ct_attr is not None else 0
            if current_subgraph not in subgraph_compute_types:
                subgraph_compute_types[current_subgraph] = set()
            subgraph_compute_types[current_subgraph].add(compute_type)

    # Build subgraph_map:  GEMM (compute_type > 0) -> core 0,  vector only -> core 1
    num_subgraphs = max(subgraph_compute_types.keys(), default=0) + 1
    subgraph_map = {}
    for sg_id in range(num_subgraphs):
        types = subgraph_compute_types.get(sg_id, set())
        has_gemm = any(t > 0 for t in types)
        subgraph_map[str(sg_id)] = 0 if has_gemm else 1

    attribute = {"subgraph_map": subgraph_map}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(attribute, f, indent=2)


def _write_default_attribute(output_path, num_subgraphs=1):
    """Fallback: assign all subgraphs to core 0."""
    subgraph_map = {str(i): 0 for i in range(num_subgraphs)}
    attribute = {"subgraph_map": subgraph_map}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(attribute, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <onnx_path> <output_path>")
        sys.exit(1)
    generate_q32_attributes(sys.argv[1], sys.argv[2])
