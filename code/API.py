from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import argparse

# 设置参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, help='data error')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(".\\model", trust_remote_code=True)


model = AutoModel.from_pretrained(".\\model", trust_remote_code=True)

if args.p == "fp16":
    model = model.half().cuda()
elif args.p == "int8":
    model = model.half().quantize(8).cuda()
elif args.p == "int4":
    model = model.half().quantize(4).cuda()
elif args.p == "cpu":
    model = model.float()

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
  try:
    json_post = await request.json()
    return {"response": json_post.get('context'), "status": 200}
  except Exception as e:
    print(f"An error occurred: {e}")
    raise HTTPException(status_code=500, detail=str(e))


# 运行服务
if __name__ == '__main__':
  uvicorn.run(app, host='127.0.0.1', port=8257, workers=1)