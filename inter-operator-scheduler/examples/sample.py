from concurrent import futures
import grpc 
import outputTrans_pb2
import outputTrans_pb2_grpc

import numpy as np
import ios
import sys
sys.path.append("..")
import python.ios.models.vgg as v 



class Trans(outputTrans_pb2_grpc.Trans):
    def output_trans(self, request, context):
        output = 1  # 初始化output
        print("output数据传输给client")
        return outputTrans_pb2.inputData(output)


def server():
    print("test_grpc")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    outputTrans_pb2_grpc.add_TransServicer_to_server(Trans(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("grpc server start...")
    server.wait_for_termination()


def  receiveInput():
    
    channel = grpc.insecure_channel('163.143.0.101:56789')                  # 连上服务器
    # with grpc.secure_channel('163.143.0.101:56789', grpc.ssl_channel_credentials()) as channel:
    print("connect the server...")
    stub = outputTrans_pb2_grpc.TransStub(channel)
    print("test_2")
    response = stub.output_trans(outputTrans_pb2.outputData())             
    print("data trans ...")    
    input = response.idata
    return input


def sample_network():
    v = ios.placeholder(output_shape=(375, 15, 15))
    block = ios.Block(enter_node=v.node)
    v1 = ios.conv2d(block, inputs=[[v]], out_channels=375, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    # v2 = ios.conv2d(block, inputs=[[v1]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    # v3 = ios.conv2d(block, inputs=[[v2]], out_channels=375, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v2 = ios.conv2d(block, inputs=[[v1]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    out = ios.identity(block, inputs=[[v1], [v2]], is_exit=True)  # concat v1, v2, and v3
    # # out = ios.identity(block, inputs=[[v1]], is_exit=True)  # concat v1, v2, and v3
    # # out = ios.identity(block, inputs=[[v3]], is_exit=True)  # concat v1, v2, and v3
    # # out = ios.identity(block, inputs=[[v2]], is_exit=True)  # concat v1, v2, and v3


    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


# define computation graph
# graph = sample_network()
# graph = ios.models.inception_v3()

graph = v.vgg_11()

# optimize execution schedule
optimized_graph = ios.optimize(graph, batch_size=1, opt_type='dp_parallel', compute_weight=True)

# measure latency
graph.sequential_schedule()
seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6, profile_stage=True)
print(graph)
print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6, profile_stage=True)
print(optimized_graph)
print(f'Optimized schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}')


dummy_inputs = receiveInput()             # 接收信息output

# inference on ios runtime
# dummy_inputs = np.random.randn(1, 375, 15, 15)
output = ios.ios_runtime.graph_inference(optimized_graph, batch_size=1, input=dummy_inputs)
# print("test_1:",output)
# print("test_2:",np.ndarray.reshape(output))






