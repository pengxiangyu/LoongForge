from core.orchestrator.graph import ExecutionGraph, Node
from core.orchestrator.invalidation import invalidation_set


def _pipeline(
    *,
    anatomy_hash: str = "v1",
    extra_leaf: bool = False,
    scaffold_dep: tuple[str, ...] = ("anatomy",),
) -> ExecutionGraph:
    nodes: list[Node] = [
        Node(id="intake", stage="intake", content_hash="v1"),
        Node(id="anatomy", stage="anatomy", depends_on=("intake",), content_hash=anatomy_hash),
        Node(id="scaffold", stage="scaffold", depends_on=scaffold_dep, content_hash="v1"),
        Node(id="migrate", stage="migrate", depends_on=("scaffold",), content_hash="v1"),
        Node(id="align", stage="align", depends_on=("migrate",), content_hash="v1"),
        Node(id="side_audit", stage="align", depends_on=("intake",), content_hash="v1"),
    ]
    if extra_leaf:
        nodes.append(
            Node(id="perf_opt", stage="align", depends_on=("align",), content_hash="v1")
        )
    return ExecutionGraph(nodes, graph_id="embodied")


def test_identical_graphs_produce_empty_invalidation_set() -> None:
    old_graph = _pipeline()
    new_graph = _pipeline()

    assert invalidation_set(old_graph, new_graph) == set()


def test_content_hash_change_propagates_transitively_without_touching_siblings() -> None:
    old_graph = _pipeline(anatomy_hash="v1")
    new_graph = _pipeline(anatomy_hash="v2")

    invalidated = invalidation_set(old_graph, new_graph)
    assert invalidated == {"anatomy", "scaffold", "migrate", "align"}
    assert "intake" not in invalidated
    assert "side_audit" not in invalidated


def test_explicit_content_changed_and_new_nodes_invalidate_downstream() -> None:
    old_graph = _pipeline()
    new_graph = _pipeline(extra_leaf=True)

    invalidated = invalidation_set(
        old_graph,
        new_graph,
        content_changed=["migrate"],
    )
    assert invalidated == {"migrate", "align", "perf_opt"}


def test_dependency_edge_change_invalidates_node_and_descendants() -> None:
    old_graph = _pipeline(scaffold_dep=("anatomy",))
    new_graph = _pipeline(scaffold_dep=("intake", "anatomy"))

    invalidated = invalidation_set(old_graph, new_graph)
    assert invalidated == {"scaffold", "migrate", "align"}


def test_group_content_changed_invalidates_children_and_downstream() -> None:
    old_graph = ExecutionGraph(
        [
            Node(id="intake", stage="intake"),
            Node(
                id="impl_group",
                stage="group",
                depends_on=("intake",),
                children=(
                    Node(id="scaffold", stage="scaffold"),
                    Node(id="migrate", stage="migrate", depends_on=("scaffold",)),
                ),
            ),
            Node(id="align", stage="align", depends_on=("impl_group",)),
        ]
    )
    new_graph = ExecutionGraph(
        [
            Node(id="intake", stage="intake"),
            Node(
                id="impl_group",
                stage="group",
                depends_on=("intake",),
                children=(
                    Node(id="scaffold", stage="scaffold"),
                    Node(id="migrate", stage="migrate", depends_on=("scaffold",)),
                ),
            ),
            Node(id="align", stage="align", depends_on=("impl_group",)),
        ]
    )

    invalidated = invalidation_set(
        old_graph,
        new_graph,
        content_changed={"impl_group"},
    )
    assert invalidated == {"scaffold", "migrate", "align"}
