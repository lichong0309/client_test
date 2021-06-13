import socket
def server():
    server = socket.socket()
    server.bind(('0.0.0.0',80))
    server.listen()
    sock,addr = server.accept()
    data = ""                   
    while True:
        tmp_data = sock.recv(1024)
        if tmp_data:
            data += tmp_data.decode("utf8")   # 解码
        else:
            break
    print('%s发送的内容：%s'%(addr[0],data))
    sock.close()
    tmp_data = list(tmp_data)                   # src转list
    return tmp_data