import socket
def client(sendServer):
    # sendServer  = str(sendServer)                   # 转成字符串
    client = socket.socket()
    client.connect(('163.143.0.101',80))
    client.send(sendServer.encode("utf8"))                  # 编码成utf-8
    client.close()