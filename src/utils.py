"""
工具模块：路径配置、日志系统、计时器装饰器
"""
import logging
import time
import functools
from pathlib import Path

# ============================================================
# 项目根目录（utils.py 位于 src/，父目录即为项目根）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据路径
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"

# 输出路径
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"


def ensure_dirs():
    """确保所有数据/输出目录存在"""
    dirs = [
        DATA_RAW, DATA_PROCESSED, DATA_FEATURES,
        OUTPUT_FIGURES, OUTPUT_TABLES, OUTPUT_MODELS,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "stock_quant") -> logging.Logger:
    """
    创建并配置 logger 实例
    参数:
        name: logger 名称
    返回:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def timer(func):
    """
    计时器装饰器，自动记录函数执行耗时
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("stock_quant")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("[计时] %s 执行耗时: %.2f 秒", func.__name__, elapsed)
        return result
    return wrapper
