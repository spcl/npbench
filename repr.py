import dace
from dace.sdfg.analysis.cutout import SDFGCutout

sdfg = dace.SDFG.from_file("aopt.sdfg")

map_entries = []
for state in sdfg.states():
    print(state)
    for node in state.nodes():
        print(node)
        if isinstance(node, dace.nodes.MapEntry):
            map_entries.append(node)

i=0
for me in map_entries:
    nodes = state.all_nodes_between(me, state.exit_node(me))
    nl = [me] + list(nodes) + [state.exit_node(me)]
    ce = SDFGCutout.singlestate_cutout(state, *nl)
    print(ce)
    ce.save(f"ce{i}.sdfg")
    i+=1