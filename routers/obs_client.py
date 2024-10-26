# 阅读该网址 https://support.huaweicloud.com/sdk-python-devg-obs/obs_22_0400.html
# 安装sdk pip install esdk-obs-python --trusted-host pypi.org

from obs import ObsClient
from obs import PutObjectHeader, GetObjectHeader
import traceback
import os

ak = "IZ96BHBOQCLE3R4M8N2F"
sk = "YppiRBPqirMpgDw9hJuNrLhLy6hwiQylnWwXUJPg"

#  server填写Bucket对应的Endpoint, 这里以华北-北京四为例，其他地区请按实际情况填写。
server = "obs.cn-north-4.myhuaweicloud.com"
obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
# 推荐通过环境变量获取AKSK，这里也可以使用其他外部引入方式传入，如果使用硬编码可能会存在泄露风险。
# 您可以登录访问管理控制台获取访问密钥AK/SK，获取方式请参见https://support.huaweicloud.com/usermanual-ca/ca_01_0003.html。
# ak = os.getenv("AccessKeyID")
# sk = os.getenv("SecretAccessKey")


def obs_upload_file(file_path:str, dir:str):

    # 创建obsClient实例
    # 如果使用临时AKSK和SecurityToken访问OBS，需要在创建实例时通过security_token参数指定securityToken值
    try:
        # ！！！！！！！！！！！注意，上传对象的附加头域，支持公开读，否则链接无法直接访问
        headers = PutObjectHeader(acl="public-read")
        bucketName = "zoomglass"
        # 【可选】待上传对象的MIME类型
        # headers.contentType = "image/png"
        # 对象名，即上传后的文件名
        # objectKey = "test/test.txt"
        # 待上传文件/文件夹的完整路径，如aa/bb.txt，或aa/
        # file_path = "./test.txt"
        # 上传文件的自定义元数据
        metadata = {"meta1": "value1", "meta2": "value2"}
        # 文件上传
        objectKey = f"models/{dir}/{os.path.basename(file_path)}"
        resp = obsClient.putFile(bucketName, objectKey, file_path, metadata, headers)
        # 返回码为2xx时，接口调用成功，否则接口调用失败
        if resp.status < 300:
            print("Put File Succeeded")
            return f"https://zoomglass.obs.cn-north-4.myhuaweicloud.com/{objectKey}"
            
        else:
            print("Put File Failed", resp.status)
            print("requestId:", resp.requestId)
            print("errorCode:", resp.errorCode)
            print("errorMessage:", resp.errorMessage)
    except:
        print("Put File Failed")
        print(traceback.format_exc())
        raise Exception("obs Put File Failed")

def obs_download_file(file_path:str, dir:str):
    bucketName = "zoomglass"
    objectKey = f"models/{dir}/{os.path.basename(file_path)}"
    downloadPath = file_path
    try:
        headers = GetObjectHeader()
        resp = obsClient.getObject(bucketName, objectKey, downloadPath, headers=headers)
        if resp.status < 300:
            print("Download File Succeeded")
        else:
            print("Download File Failed", resp.status)
            print("requestId:", resp.requestId)
            print("errorCode:", resp.errorCode)
            print("errorMessage:", resp.errorMessage)
    except:
        print("Download File Failed")
        print(traceback.format_exc())
        raise Exception("obs Download File Failed")

