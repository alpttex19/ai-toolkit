from routers import swagger_monkey_patch

from fastapi import applications
applications.get_swagger_ui_html  = swagger_monkey_patch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
app = FastAPI() # 配置 CORS
app.add_middleware( CORSMiddleware, allow_origins=["*"], # 允许的源 
                    allow_credentials=True, allow_methods=["*"], # 允许的 HTTP 方法 
                    allow_headers=["*"], # 允许的 HTTP 头 
                    )

import interface_gen, interface_train
app.include_router(interface_train.router)
app.include_router(interface_gen.router)


# 之后可以使用
# app.include_router(
#     admin.router,
#     prefix="/admin",
#     tags=["admin"],
#     dependencies=[Depends(get_token_header)],
#     responses={418: {"description": "I'm a teapot"}},
# )


@app.get("/")
def root():
    return {"message": "Hello Bigger Applications!"}

if __name__ == '__main__':
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=6006, type=int)
    uvicorn.run(app, host="127.0.0.1", port=6006)
