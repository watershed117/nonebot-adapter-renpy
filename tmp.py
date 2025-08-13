import base64

def byte_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

with open(r"C:\Users\water\Desktop\renpy\Ushio_Noa\game\images\background\sports meet\kivotosstadium2_sunset.jpg","rb") as f:
    data = f.read()

base64_data = byte_to_base64(data)

decode_data = base64.b64decode(base64_data.encode("utf-8"))

with open(r"C:\Users\water\Desktop\tmp.jpg","wb") as f:
    f.write(decode_data)