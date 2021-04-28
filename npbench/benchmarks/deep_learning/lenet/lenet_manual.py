import numpy as np
import dace as dace

from dace.transformation.auto_optimize import auto_optimize
from dace.transformation.subgraph import ReduceExpansion
from dace.transformation.dataflow import MapCollapse, MapExpansion
from dace.transformation.interstate import LoopToMap

N, H, W, C_before_fc1, S0, S1, S2, S3, S4, S5 = (dace.symbol(
    s, dtype=dace.int64) for s in ('N', 'H', 'W', 'C_before_fc1', 'S0', 'S1',
                                   'S2', 'S3', 'S4', 'S5'))


@dace.program
def relu2(x: dace.float32[S0, S1]):
    return np.maximum(x, 0)


@dace.program
def relu4(x: dace.float32[S0, S1, S2, S3]):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@dace.program
def conv2d(input: dace.float32[S0, S1, S2, S3], weights: dace.float32[S4, S4,
                                                                      S3, S5]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dace.map[0:H-K+1, 0:W-K+1]:
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(
                input[:, i:i + S4, j:j + S4, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )
            # # TODO: View operations are needed
            # output[:, i, j, :] = np.sum(
            #     np.reshape(input[:, i:i+S4, j:j+S4, :], (S0, S4, S4, S3, 1)) *
            #     np.reshape(weights, (1, S4, S4, S3, S5)),
            #     axis=(1, 2, 3),
            # )

    return output


# LeNet-5 Convolutional Neural Network (inference mode)
@dace.program
def lenet5_ca(
    input: dace.float32[N, H, W, 1],
    conv1: dace.float32[5, 5, 1, 6],
    conv1bias: dace.float32[6],
):
    # x = relu(conv2d(input, conv1) + conv1bias)
    # x = maxpool2d(x)
    # x = relu(conv2d(x, conv2) + conv2bias)
    # x = maxpool2d(x)
    # x = np.reshape(x, (N, C_before_fc1))
    # x = relu(x @ fc1w + fc1b)
    # x = relu(x @ fc2w + fc2b)
    return relu4(conv2d(input, conv1) + conv1bias)


def get_args():
    N, H, W = 64, 512, 512
    symbols = {'H': H, 'W': W}
    inputs = {
        'input': np.random.rand(N, H, W, 1).astype(np.float32),
        'conv1': np.random.rand(5, 5, 1, 6).astype(np.float32),
        'conv1bias': np.random.rand(6).astype(np.float32)
    }
    return {**symbols, **inputs}


args = get_args()
sdfg = lenet5_ca.to_sdfg()
sdfg.specialize({'N': 64})

# get rid of view node
for sd in sdfg.all_sdfgs_recursive():
    dace.sdfg.propagation.propagate_states(sd)
strict_transformations = dace.transformation.strict_transformations()
sdfg.apply_transformations_repeated([LoopToMap] + strict_transformations,
                                    strict=True,
                                    validate=False,
                                    validate_all=True)
sdfg.apply_transformations(LoopToMap, strict=False)
sdfg.apply_gpu_transformations()
graph = sdfg.nodes()[0]
sdfg.save('baseline.sdfg')
r1 = sdfg(**args)
print(np.linalg.norm(r1))

for node in graph.nodes():
    if isinstance(node, dace.libraries.standard.nodes.Reduce):
        ReduceExpansion.apply_to(sdfg, _reduce=node)
sdfg.save('expanded.sdfg')
for node in graph.nodes():
    if isinstance(node, dace.sdfg.nodes.AccessNode):
        if isinstance(sdfg.data(node.data), dace.data.View):
            map_entry = graph.in_edges(node)[0].src
            reduce_node = graph.in_edges(node)[0].src

            for e in graph.memlet_tree(graph.in_edges(node)[0]):
                e.data.data = 'conv2d_ret_0'
                e.data.subset = dace.subsets.Range(
                    (e.data.subset.ranges[0], ('i', 'i', 1), ('j', 'j', 1),
                     e.data.subset.ranges[1]))

            edge = graph.add_edge(
                graph.in_edges(node)[0].src,
                graph.in_edges(node)[0].src_conn,
                graph.out_edges(node)[0].dst,
                graph.out_edges(node)[0].dst_conn,
                graph.out_edges(node)[0].data)
            graph.remove_node(node)

sdfg.save('before_optimization.sdfg')
print("-------------")
auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
print("MAP COLLAPSE")
sdfg.apply_transformations(MapCollapse)

sdfg.apply_transformations_repeated(MapExpansion)
for _ in range(2):
    for node in graph.nodes():
        if isinstance(
                node,
                dace.sdfg.nodes.AccessNode) and node.label == 'gpu_input':
            o1 = graph.out_edges(node)[0].dst
            o2 = graph.out_edges(o1)[0].dst
            print(o1, o2)
            MapCollapse.apply_to(sdfg,
                                 _outer_map_entry=o1,
                                 _inner_map_entry=o2)
            break

for node in sdfg.nodes()[0].nodes():
    if isinstance(node, dace.sdfg.nodes.MapEntry) and 'inner' in node.label:
        print("ASSIGN SEQUENTIAL")
        node.schedule = dace.dtypes.ScheduleType.Sequential
sdfg._name = 'after'
sdfg.save('after_optimization.sdfg')

r2 = sdfg(**get_args())
print(np.linalg.norm(r2))
