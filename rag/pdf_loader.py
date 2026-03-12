"""
PDF 解析模块
职责：
  - 提取 PDF 中的文字内容（保留结构，过滤页眉页脚噪声）
  - 提取 PDF 中内嵌的图片
  - 将图片发送给 Kimi 视觉模型生成文字描述
  - 将文字和图片描述切块后返回，供存入向量库
"""

import base64
import io
import logging
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import KIMI_API_KEY, KIMI_BASE_URL, KIMI_VISION_MODEL

logger = logging.getLogger(__name__)

# ── 噪声过滤规则 ────────────────────────────────────────────
# 页眉页脚通常出现在页面顶部/底部固定区域
PAGE_HEADER_RATIO = 0.08   # 页面高度前 8% 视为页眉区域
PAGE_FOOTER_RATIO = 0.92   # 页面高度后 8% 视为页脚区域

# 过短的文字块通常是页码、装饰性文字
MIN_TEXT_LENGTH = 10

# ── 切块配置 ────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class PDFLoader:
    """
    PDF 图文混合内容解析器。

    使用流程：
        loader = PDFLoader()
        chunks = loader.load(pdf_path)
        # chunks 是 List[Document]，直接存入 ChromaDB
    """

    def __init__(self) -> None:
        # 视觉模型：复用主模型（kimi-k2-turbo-preview 原生支持图片输入）
        self._vision_llm = ChatOpenAI(
            model=KIMI_VISION_MODEL,
            api_key=SecretStr(KIMI_API_KEY),
            base_url=KIMI_BASE_URL,
            temperature=0.3,
            max_completion_tokens=500,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ""],
        )

    # ── 主入口 ──────────────────────────────────────────────

    def load(self, pdf_path: str) -> List[Document]:
        """
        解析整个 PDF，返回可直接存入向量库的 Document 列表。
        每个 Document 的 metadata 包含：
          - source: 文件路径
          - page: 页码（从1开始）
          - content_type: "text" 或 "image"
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

        logger.info("✦ 开始解析 PDF：%s", pdf_path)
        doc = fitz.open(pdf_path)

        all_chunks: List[Document] = []
        total_images = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_label = page_num + 1

            # 1. 提取并清洗文字
            text_chunks = self._extract_text(page, page_label, str(path))
            all_chunks.extend(text_chunks)

            # 2. 提取图片并生成描述
            image_chunks = self._extract_images(doc, page, page_label, str(path))
            all_chunks.extend(image_chunks)
            total_images += len(image_chunks)

        doc.close()
        logger.info(
            "✦ PDF 解析完成：%s，共 %d 个文字 chunk，%d 张图片",
            path.name, len(all_chunks) - total_images, total_images
        )
        return all_chunks

    # ── 文字提取 ────────────────────────────────────────────

    def _extract_text(
        self, page: fitz.Page, page_num: int, source: str
    ) -> List[Document]:
        """
        提取单页文字，过滤页眉页脚和噪声，切块后返回。
        """
        page_height = page.rect.height
        header_threshold = page_height * PAGE_HEADER_RATIO
        footer_threshold = page_height * PAGE_FOOTER_RATIO

        blocks = page.get_text("blocks")  # 返回 (x0, y0, x1, y1, text, block_no, block_type)
        clean_texts = []

        for block in blocks:
            x0, y0, x1, y1, text, _, block_type = block

            # 只处理文字块（block_type=0），跳过图片块（block_type=1）
            if block_type != 0:
                continue

            text = text.strip()

            # 过滤页眉页脚（按位置）
            if y1 < header_threshold or y0 > footer_threshold:
                logger.debug("过滤页眉/页脚：page=%d y0=%.1f y1=%.1f", page_num, y0, y1)
                continue

            # 过滤太短的碎片
            if len(text) < MIN_TEXT_LENGTH:
                continue

            # 过滤纯数字（页码）
            if re.fullmatch(r'\d+', text):
                continue

            clean_texts.append(text)

        if not clean_texts:
            return []

        full_text = "\n\n".join(clean_texts)

        # 切块
        raw_docs = [Document(page_content=full_text, metadata={
            "source": source,
            "page": page_num,
            "content_type": "text",
        })]
        return self._splitter.split_documents(raw_docs)

    # ── 图片提取 ────────────────────────────────────────────

    def _extract_images(
        self, doc: fitz.Document, page: fitz.Page, page_num: int, source: str
    ) -> List[Document]:
        """
        提取单页所有图片，用视觉模型生成描述，返回 Document 列表。
        装饰性小图（尺寸过小）会被跳过。
        """
        image_list = page.get_images(full=True)
        chunks = []

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
            except Exception as e:
                logger.warning("图片提取失败 page=%d index=%d: %s", page_num, img_index, e)
                continue

            # 跳过装饰性小图（宽或高小于 80px）
            if width < 80 or height < 80:
                logger.debug("跳过小图 page=%d size=%dx%d", page_num, width, height)
                continue

            # 查找图片附近的图注文字
            caption = self._find_caption(page, xref)

            # 调用视觉模型理解图片
            description = self._describe_image(image_bytes, img_ext, caption, page_num)
            if not description:
                continue

            # 图注 + 视觉描述合并为一个 chunk
            content = description
            if caption:
                content = f"{caption}\n{description}"

            chunks.append(Document(
                page_content=content,
                metadata={
                    "source": source,
                    "page": page_num,
                    "content_type": "image",
                    "image_size": f"{width}x{height}",
                }
            ))

        return chunks

    def _find_caption(self, page: fitz.Page, xref: int) -> str:
        """
        在图片附近查找图注文字（匹配"图X"、"Figure X"等格式）。
        """
        caption_patterns = [
            r'图\s*\d+',
            r'Figure\s*\d+',
            r'Fig\.\s*\d+',
            r'插图\s*\d*',
        ]
        full_text: str = page.get_text()  # type: ignore[assignment]
        for pattern in caption_patterns:
            match = re.search(pattern + r'[^\n]*', full_text)
            if match:
                return match.group().strip()
        return ""

    def _describe_image(
        self, image_bytes: bytes, ext: str, caption: str, page_num: int
    ) -> str:
        """
        调用 Kimi 视觉模型，将图片转换为文字描述。
        """
        try:
            b64 = base64.b64encode(image_bytes).decode()
            media_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"

            context_hint = f"（图注：{caption}）" if caption else ""
            prompt = (
                f"请详细描述这张图片的内容{context_hint}，"
                "提取其中所有文字信息、图表数据和关键视觉元素。"
                "用中文回答，描述要具体且信息完整。"
            )

            response = self._vision_llm.invoke([{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }])

            description = str(response.content).strip()
            logger.info("✦ 图片描述完成 page=%d: %s...", page_num, description[:30])
            return f"[图片内容描述] {description}"

        except Exception as e:
            logger.warning("图片描述失败 page=%d: %s", page_num, e)
            return ""