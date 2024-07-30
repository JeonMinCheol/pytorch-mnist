import grpc
from concurrent import futures
import protos.message_pb2_grpc as pb2_grpc
import protos.message_pb2 as pb2
import torch
import io


class sendParamsService(pb2_grpc.sendParamsServicer):

    def __init__(self, *args, **kwargs):
        self.models = list() # (시점, 값, 디바이스 개수)

    def GetServerResponse(self, request, context):
        model, device = io.BytesIO(request.model), request.device # 들어오면 자동으로 역직렬화
        model = torch.load(model)
        self.models.append(model)
        
        result = {'message': device, 'received': True}
        print(result)
        
        return pb2.MessageResponse(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_sendParamsServicer_to_server(sendParamsService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()