import cloudpickle
import json
import binascii
import sys
sys.path.append('/home/ankur/dev/apps/ML/OCR/aster.pytorch.prefect')
#with open("/home/ankur/.prefect/flows/ocr-pipeline/2021-02-04t20-22-42-461103-00-00", "rb") as f:
#     flow = cloudpickle.load(f)

with open("/home/ankur/.prefect/flows/ocr-pipeline/2021-03-03t21-33-47-382603-00-00", "rb") as f:
    f_ = f.read()
    info = json.loads(f_.decode("utf-8"))
    flow_bytes = binascii.a2b_base64(info["flow"])
    try:
        flow = cloudpickle.loads(flow_bytes)
        print('ok')
    except Exception as exc:
        parts = ["An error occurred while unpickling the flow:", f"  {exc!r}"]


print('done')