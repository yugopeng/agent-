"""
文件处理工具模块
为 RAG 知识库入库流程提供：文件MD5指纹计算、目录文件过滤、PDF/TXT文档加载
"""
import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str) -> str | None:
    """
    计算文件的MD5十六进制字符串，作为文件内容指纹用于知识库去重
    :param filepath: 待计算MD5的文件绝对路径
    :return: MD5十六进制字符串；文件不存在、不是文件或读取失败时返回None
    """
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()

    chunk_size = 4096       # 4KB分片读取，避免大文件一次性载入爆内存
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取，文本模式会做编码转换导致哈希不一致
            # 海象运算符：读一块、处理一块，直到文件读空为止
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str, ...]) -> tuple[str, ...]:
    """
    遍历文件夹，返回后缀在允许类型范围内的文件绝对路径元组
    :param path: 待扫描的文件夹路径
    :param allowed_types: 允许的文件后缀元组，如(".txt", ".pdf")
    :return: 匹配文件的绝对路径元组；path不是文件夹时原样返回allowed_types
    """
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(filepath: str, passwd: str | None = None) -> list[Document]:
    """
    使用PyPDFLoader加载PDF文件，按页拆分为Document
    :param filepath: PDF文件路径
    :param passwd: 加密PDF的打开密码，未加密文件无需传参
    :return: 解析出的Document列表，每页对应一个Document
    """
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    """
    使用TextLoader加载文本文件，全文作为单个Document
    :param filepath: 文本文件路径
    :return: 包含文本内容的Document列表
    """
    return TextLoader(filepath, encoding="utf-8").load()
