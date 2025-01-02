import json
import re
import tempfile
import zipfile
from pathlib import Path
from random import shuffle
from typing import TypeAlias, List, Self, Dict, Optional, Tuple

import fitz
from loguru import logger

from crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
from toolbox import promote_file_to_downloadzone

Briefing: TypeAlias = str
WrittenChap: TypeAlias = str


class ChapterOutline:
    """
    用于处理文章章节的提纲
    """

    AFFIRMATIVE = "是相关的"
    REJECT = "没有关系"
    MAX_WORKER = 4
    TITLE = ""

    def __init__(self, content: str, llm_kwargs, chatbot, references: List[Path] = None):
        self.content = content
        self.references = references or []
        self._llm_kwargs = llm_kwargs
        self._chatbot = chatbot
        self._role = f'You are expert about how to write a paper about "{self.TITLE}" and how to choose most appropriate references.'

    @property
    def chap_header(self) -> str:
        """
        获取章节标题
        """
        segs = self.content.split("\n")
        if len(segs) > 1:
            return (
                segs[0]
                .replace("\n", "")
                .replace("\r", "")
                .replace("\t", "")
                .replace("*", "")
                .replace(":", "")
                .replace("?", "")
                .replace("<", "")
                .replace(">", "")
                .replace("/", "")
                .replace("\\", "")
            )
        else:
            return self.content[:20]

    def set_references(self, references: List[Path]) -> Self:
        """
        设置文献综述的路径
        """
        self.references.clear()
        self.references.extend(references)
        return self

    @staticmethod
    def load_references(ref_paths: List[Path]) -> List[Briefing]:
        """
        读取文献综述的内容
        """
        return [p.read_text("utf-8") for p in ref_paths]

    def load_self_references(self) -> List[Briefing]:
        """
        读取文献综述的内容
        """
        return self.load_references(self.references)

    def relation_asm(self, briefing: Briefing, threshold: int = 88) -> str:
        """
        生成用于评估文献综述与论文提纲相关性的 ASM 任务。
        """
        return (
            f"文献综述:\n{briefing}\n\n"
            f"请根据以下标准判断上面这篇文献综述是否适合作为题目是“{self.TITLE}”的论文提纲撰写参考材料：\n\n"
            f"1. 综述中的核心工作是否对我们下面的章节编写有参考价值。\n"
            f"2. 综述中的主要方法是否可以借鉴过来实现我们自己的接下来的章节。\n"
            f"3. 综述中的主要结论是否可以帮助我们在接下来的章节做出一些判断。\n"
            f"4. 综述中的对于方法的局限性是否可以在我们接下来这一章里面得到改进或者优化\n"
            f"5. 综述中的图或者表是不是可以在我们这里拿来引用，用作辅助说明我们的论点\n"
            f"论文提纲:\n{self.content}\n\n"
            f"如果综述与提纲有高于或等于{threshold}%的相关度，可以作为引文插入到提纲内的章节里面，回复关键字“{self.AFFIRMATIVE}”。\n"
            f"如果综述与提纲有低于{threshold}%的相关度，作为引文插入到提纲内的章节里面会让衔接显得奇怪并且没有逻辑联系，回复关键字“{self.REJECT}”。\n"
            f"你仅需要回复上述【{self.AFFIRMATIVE}/{self.REJECT}】两个中关键字之一，不要带任何其他的说明，你给出的任何额外解释或者理由与原因均会被视为无效输入。"
        )

    def write_batch_asm(self) -> str:
        """
        生成用于撰写文章内容的ASM任务指令。
        """
        ref_materials = self.load_self_references()
        asm_ref_material = "\n\n".join([f"[{i}]: {ref}" for i, ref in enumerate(ref_materials)])

        return f"""
        {asm_ref_material}
        
        总共综述了{len(ref_materials)}篇相关文献。
        现在，请根据上述文献，为题为“{self.TITLE}”的论文撰写一个完整的章节，基于以下提纲:
        
        {self.content}
        
        要求：
        1. 融合全部{len(ref_materials)}篇文献的内容，确保无遗漏。
        2. 结合文献综述扩展提纲内容，避免直接复制粘贴，适当改写以丰富章节。
        3. 引用文献时，使用实际作者姓氏及年份，遵循学术引用规范。例如，“作者1，作者2，作者3等人（2013）”，对于两位作者：“作者1与作者2（2013）”，单个作者：“作者1（2013）”。多篇文献引用时，如“作者1（2019），作者2（2024）”。
        4. 若有图表可作为辅助说明，参考并引用，格式为“作者1，作者2，作者3等人（2016）的图-叶片疲劳曲线所示...”，仅需说明引用位置，多媒体文件插入由我完成。
        5. 遵循提供的提纲结构，保持章节编号不变。
        6. 输出应为纯文本，不使用Markdown语法或其他标记语言，标题序号遵循x x.y x.y.z格式。
        
        请直接开始撰写，无需额外说明或确认。
        """

    def _write_iter_asm_inner(self, grouped: List[Path], written_article_place_holder: str) -> str:
        """
        生成用于增量撰写文章内容的 ASM 任务指令。
        """

        # 加载引用文献
        grouped = self.load_references(grouped)

        # 构建参考文献字符串
        asm_ref_material = "\n\n".join([f"[{i}]: {ref_material}" for i, ref_material in enumerate(grouped)])

        # 指令模板
        instruction_template = (
            f"{asm_ref_material}\n\n"
            "任务说明：\n"
            f"我的论文的已完成部分：\n{written_article_place_holder}\n\n"
            f"正在编写题目为“{self.TITLE}”的论文，根据提供的提纲和已完成的部分，将上方所有的{len(grouped)}篇参考文献综述作为引文全部插入到章节中。\n"
            "要求：\n"
            "- 不得遗漏任何已经存在的引用文献，保持原有引用完整性。\n"
            "- 章节编号严格依照提纲，不得私自增减章节或者子章节。如果发现我给出的已完成部分里面有不在我给出的提纲存在的章节你需要把它们给删除掉\n"
            "- 行文流畅，逻辑严谨，丰富并完善章节内容。\n"
            "- 引用格式：根据作者数量正确使用引文格式，引用文献时，使用实际作者姓氏及年份，遵循学术引用规范。"
            "例如，“作者1，作者2，作者3等人（2013）”，"
            "对于两位作者：“作者1与作者2（2013）”，"
            "单个作者：“作者1（2013）”。"
            "多篇文献引用时，如“作者1（2019），作者2（2024）”。\n"
            "- 可引用参考文献中的图标辅助说明，具体来说你在文中把原图或表的题注写到文内，写的时候不要翻译源题注的文本，最后我会根据你写的题注的位置来插入你所引用的图表。一定要多引用图，图包含的信息更多，可以让论文变得更加优质可读。\n"
            "- 输出为纯文本，不使用Markdown语法或其他标记语言，标题序号遵循x x.y x.y.z格式。\n"
            "- 无需开头问候或结尾总结。\n\n"
            f"这是本次任务你需要完成撰写的提纲：\n{self.content}\n\n"
            "请按照上述要求完成论文章节的撰写。"
        )

        return instruction_template

    def _write_iter_missing_ref_asm_inner(self, grouped: List[Path], written_article_place_holder: str) -> str:
        """
        生成用于增量撰写文章内容的 ASM 任务指令。
        """

        segs = [FileNameSegmentation(p) for p in grouped]
        labels = [f"{seg.authors}等人（{seg.year}）" for seg in segs]

        # 加载引用文献
        grouped = self.load_references(grouped)

        # 构建参考文献字符串
        asm_ref_material = "\n\n".join([f"[{i}]: {ref_material}" for i, ref_material in enumerate(grouped)])

        # 指令模板
        instruction_template = (
            f"{asm_ref_material}\n\n"
            "任务说明：\n"
            f"我的论文的已完成部分：\n{written_article_place_holder}\n\n"
            f"正在编写题目为“{self.TITLE}”的论文，根据提供的提纲和已完成的部分，将上方所有的{len(grouped)}篇参考文献综述作为引文全部插入到章节中。\n"
            "要求：\n"
            "- 不得遗漏任何已经存在的引用文献，保持原有引用完整性。\n"
            "- 章节编号严格依照提纲，不得私自增减章节或者子章节。如果发现我给出的已完成部分里面有不在我给出的提纲存在的章节你需要把它们给删除掉\n"
            "- 行文流畅，逻辑严谨，丰富并完善章节内容。\n"
            "- 引用格式：根据作者数量正确使用引文格式，引用文献时，使用实际作者姓氏及年份，遵循学术引用规范。"
            "例如，“作者1，作者2，作者3等人（2013）”，"
            "对于两位作者：“作者1与作者2（2013）”，"
            "单个作者：“作者1（2013）”。"
            "多篇文献引用时，如“作者1（2019），作者2（2024）”。\n"
            "- 可引用参考文献中的图标辅助说明，具体来说你在文中把原图或表的题注写到文内，写的时候不要翻译源题注的文本，最后我会根据你写的题注的位置来插入你所引用的图表。一定要多引用图，图包含的信息更多，可以让论文变得更加优质可读。\n"
            "- 输出为纯文本，不使用Markdown语法或其他标记语言，标题序号遵循x x.y x.y.z格式。\n"
            "- 无需开头问候或结尾总结。\n\n"
            "请按照上述要求完成论文章节的撰写。"
            f"这是本次任务你需要完成撰写的提纲：\n{self.content}\n\n"
            "注意在我给出的论文已完成部分中就是缺少了下面这些人的论文的引用，你需要在我给出的论文中根据他们论文的工作内容找到合适的位置插入它们作为引用。\n"
            f"- {'\n- '.join(labels)}"
        )

        return instruction_template

    def write_iter_asm(
        self, grouped_paths: List[List[Path]], written_article_place_holder: str = "__written__"
    ) -> List[str]:
        """
        生成用于增量撰写文章内容的 ASM 任务
        """

        return [self._write_iter_asm_inner(grouped, written_article_place_holder) for grouped in grouped_paths]

    def content_group_by(self, group_size: int = 4) -> List[List[Briefing]]:
        """
        将文献综述分组
        """
        ref_materials: List[Briefing] = self.load_self_references()
        grouped_ref_materials = [ref_materials[i : i + group_size] for i in range(0, len(ref_materials), group_size)]
        return grouped_ref_materials

    def path_group_by(self, group_size: int = 4) -> List[List[Path]]:
        """
        将文献综述分组
        """
        grouped_ref_materials = [
            self.references[i : i + group_size] for i in range(0, len(self.references), group_size)
        ]
        return grouped_ref_materials

    def update_related_references(
        self,
        briefings_path: List[Path],
        pre_defined_reference: Dict[str, List[str]] = None,
        relativity_threshold: int = 88,
    ) -> Self:
        """
        用于处理文献综述与提纲关系的 ASM 任务
        """
        logger.info(f"开始处理{self.chap_header}的文献综述,过滤出符合条件的文献")

        pre_defined_reference = pre_defined_reference or {}
        briefings_names = [briefing.name for briefing in briefings_path]
        if self.chap_header in pre_defined_reference and all(
            [ref in briefings_names for ref in pre_defined_reference[self.chap_header]]
        ):
            logger.info(f"使用预设的文献综述关系， 共{len(pre_defined_reference[self.chap_header])}篇")
            self.set_references(
                list(filter(lambda x: x.name in pre_defined_reference[self.chap_header], briefings_path))
            )
        else:
            briefings = self.load_references(briefings_path)
            res = yield from (
                request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
                    handle_token_exceed=False,
                    llm_kwargs=self._llm_kwargs,
                    chatbot=self._chatbot,
                    inputs_array=[
                        self.relation_asm(briefing, threshold=relativity_threshold) for briefing in briefings
                    ],
                    inputs_show_user_array=[briefing.split("\n")[0] for briefing in briefings],
                    history_array=[[]] * len(briefings),
                    sys_prompt_array=[self._role] * len(briefings),
                    max_workers=self.MAX_WORKER,
                )
            )
            self.set_references(self.check_pass(briefings_path, res[1::2]))
        return self

    def write_iter_unused_asm(
        self,
        written,
        ref_materials: List[Path],
    ) -> Optional[str]:
        """
        构造未使用的文献综述的 ASM 任务
        """
        unused = self.get_incited_brute(ref_materials, written)
        if not unused:
            return None
        shuffle(unused)
        written_article_place_holder: str = "__written__"
        return self._write_iter_missing_ref_asm_inner(unused, written_article_place_holder).replace(
            written_article_place_holder, written
        )

    @classmethod
    def check_pass(cls, refs: List[Path], responses: List[str]) -> List[Path]:
        """
        检查文献综述与提纲关系的 ASM 任务的回答是否符合要求
        """
        assert len(refs) == len(
            responses
        ), f"The length of refs and response should be the same, ref: {len(refs)}, response: {len(responses)}"
        return list(
            map(
                lambda a: a[0],
                filter(
                    lambda x: x[1] == cls.AFFIRMATIVE or cls.AFFIRMATIVE in x[1],
                    zip(refs, responses),
                ),
            )
        )

    @staticmethod
    def get_incited(refs: List[Path], response: str, max_distance: int = 100) -> List[Path]:
        """
        Identifies references that are incorrectly cited in a given response text.

        This function searches for references that are potentially incorrectly cited based on the distance between
        the authors' names and the year in the response text. If the year appears too far from the authors' names
        or in an incorrect order, the reference is considered incorrectly cited.

        Parameters:
        refs (List[Path]): A list of file paths for the references.
        response (str): The response text in which to search for citations.
        max_distance (int): The maximum allowed distance between the authors' names and the year for a citation to be considered correct.

        Returns:
        List[Path]: A list of file paths for references that are incorrectly cited.
        """

        # Initialize a list to store the file name segmentation results
        f_segs: List[FileNameSegmentation] = [FileNameSegmentation(p) for p in refs]  # File name segmentation

        # Initialize a set to store references that are incorrectly cited
        cited_incorrectly = set()

        # Iterate through each segmented file name
        for seg in f_segs:
            if len(seg.authors_first_segment) <= 3:
                logger.debug(f"{seg.authors}|作者名字太短,检测适用效果差，跳过")
                continue
            start_index = 0
            found = False
            while True:
                # Find the position of the authors' names and year in the response text starting from the current search position
                author_index = response.find(seg.authors_first_segment, start_index)
                year_index = -1
                # If the authors' names are not found and a citation has been found before, it is considered a legal citation
                if found and author_index == -1:
                    break

                # If the authors' names are found but the year is not, it is considered an incorrect citation
                elif author_index == -1 or (year_index := response.find(seg.year, author_index)) == -1:
                    logger.debug(
                        f"{seg.authors}|未找到年份或者作者|年份：{seg.year} at {year_index}|作者：{seg.authors_first_segment} at {author_index}"
                    )
                    cited_incorrectly.add(seg)
                    break
                logger.debug(
                    f"{seg.authors}|作者{seg.authors_first_segment}位置: {author_index}, 年份{seg.year}位置: {year_index}"
                )
                # Calculate the distance between the authors' names and the year
                distance = year_index - author_index

                # If the year appears before the authors' names or the distance exceeds the maximum, it is considered an incorrect citation
                if distance > max_distance:
                    logger.debug(f"{seg.authors}|距离太远: {distance}")
                    cited_incorrectly.add(seg)
                    break

                # If the year appears after the authors' names and within the allowed distance, update the search position and continue searching
                found = True
                start_index = year_index + len(seg.year)

        logger.info(f"未找到的引用数量: {len(cited_incorrectly)}")
        # Return the list of file paths for references that are incorrectly cited
        return [f.source for f in cited_incorrectly]

    @staticmethod
    def get_incited_brute(refs: List[Path], response: str) -> List[Path]:
        """
        Identifies references that are incorrectly cited in a given response text using a brute force approach.

        This function searches for references that are potentially incorrectly cited by comparing the authors' names
        and the year in the response text with the authors' names and year in each reference. If the year appears
        in the response text but does not match the year in the reference, the reference is considered incorrectly cited.

        Parameters:
        refs (List[Path]): A list of file paths for the references.
        response (str): The response text in which to search for citations.

        Returns:
        List[Path]: A list of file paths for references that are incorrectly cited.
        """

        # Initialize a list to store the file name segmentation results
        f_segs: List[FileNameSegmentation] = [FileNameSegmentation(p) for p in refs]
        return [
            f_seg.source
            for f_seg in f_segs
            if (f_seg.year not in response and f_seg.authors_first_segment not in response)
        ]


class ContentPacker:
    """
    用于将多个文本内容打包为一个ZIP文件的工具类
    """

    def __init__(self):
        # 创建一个临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        self.files: List[Path] = []

    def add_content(self, title: str, content: str) -> Self:
        """添加带有标题的文本内容到临时目录中，并返回自身以便链式调用"""

        logger.info(f"添加内容到临时目录: {title}")
        file_path = Path(self.temp_dir.name) / f"{title}.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        self.files.append(file_path)
        return self  # 返回自身以支持链式调用

    def pack(self, output_path: str) -> Self:
        """将所有添加的内容打包为ZIP文件，并输出到指定路径"""
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in self.files:
                # 将文件添加到ZIP文件中，保持原始文件名
                zipf.write(file, arcname=file.name)
        print(f"ZIP文件已创建: {output_path}")
        return self

    def pack_and_cleanup(self, output_path: str) -> str:
        """
        打包所有内容为ZIP文件，并清理临时目录
        """
        self.pack(output_path).cleanup()
        return output_path

    def cleanup(self) -> None:
        """清理临时目录"""
        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.files.clear()


class CitationMaker:
    """
    用于生成文章引用信息的工具类
    """

    def __init__(self, ref_paths: List[Path]):
        self._ref_paths = ref_paths

    def remove_used_refs(self, used_refs: List[Path]) -> Self:
        """
        从参考文献列表中移除已经使用过的文献
        """
        self._ref_paths = list(filter(lambda x: x not in used_refs, self._ref_paths))

        return self

    @property
    def ref_path(self) -> List[Path]:
        """
        获取参考文献列表
        """
        return self._ref_paths

    def set_refs(self, refs: List[Path]) -> Self:
        """
        设置参考文献列表
        """
        self._ref_paths.clear()
        self._ref_paths.extend(refs)
        return self

    def remove_all_used_ref(self, chaps: List[ChapterOutline]) -> Self:
        """
        从参考文献列表中移除已经使用过的文献
        """
        used_refs = []
        for chap in chaps:
            used_refs.extend(chap.references)
        self.remove_used_refs(used_refs)
        return self


def dump_ref_usage_manifest(chaps: List["ChapterOutline"], all_refs: List[Path], chatbot):
    """
    将文献综述与提纲的关系保存为ZIP文件
    """
    cite_maker = CitationMaker(all_refs)
    manifest = {}

    for chap in chaps:
        manifest[chap.chap_header] = [p.name for p in chap.references]

    # 添加未使用的引用
    unused_refs = cite_maker.remove_all_used_ref(chaps)
    manifest["UNUSED"] = [p.name for p in unused_refs.ref_path]

    # 使用 tempfile 创建一个命名的临时文件
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
        json.dump(manifest, temp_file, indent=2, ensure_ascii=False)

    # 提升文件到下载区域，并指定文件名为 "citation_info.json"
    promote_file_to_downloadzone(temp_file_path, "citation_info.json", chatbot=chatbot)


def dump_final_result(chap_outlines, chatbot, gpt_res, root):
    """
    将文章内容生成为ZIP文件
    """
    packer = ContentPacker()
    for chap, resp in zip(chap_outlines, gpt_res):
        packer.add_content(f"{chap.chap_header}-write", resp)
    packer.add_content(ChapterOutline.TITLE, "\n".join(gpt_res))
    f_path = (root / (fi_name := f"{ChapterOutline.TITLE}-write.zip")).as_posix()
    packer.pack(f_path)
    logger.info(f"已经生成最终文章内容ZIP文件: {f_path}")
    out_path = promote_file_to_downloadzone(f_path, fi_name, chatbot=chatbot)
    return out_path


def dump_materials(chap_outlines, chatbot, root):
    """
    将文献综述和提纲的关系保存为ZIP文件
    """
    packer = ContentPacker()
    for chap in chap_outlines:
        packer.add_content(chap.chap_header, chap.write_batch_asm())
    packer.pack(pre_obj := (root / (f_name := f"{ChapterOutline.TITLE}.zip")).as_posix()).cleanup()
    promote_file_to_downloadzone(pre_obj, f_name, chatbot=chatbot)


def write_article(chap_outlines, chatbot, llm_kwargs, max_write_threads) -> List[str]:
    """
    生成最终的文章内容
    """

    logger.info(f"开始生成最终的文章内容")
    collections = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
        handle_token_exceed=False,
        llm_kwargs=llm_kwargs,
        chatbot=chatbot,
        inputs_array=[chap.write_batch_asm() for chap in chap_outlines],
        inputs_show_user_array=[f"Dealing with {chap.chap_header}" for chap in chap_outlines],
        history_array=[[]] * len(chap_outlines),
        sys_prompt_array=[
            f'You are expert about how to write a paper about "{ChapterOutline.TITLE}" with given references'
        ]
        * len(chap_outlines),
        max_workers=max_write_threads,
    )
    gpt_res: List[str] = collections[1::2]
    return gpt_res


def remove_markdown_syntax(content: WrittenChap) -> WrittenChap:
    """
    移除Markdown语法
    """
    return (
        content.replace("```", "")
        .replace("```plaintext", "")
        .replace("**", "")
        .replace("*", "")
        .replace("# ", "")
        .replace("## ", "")
        .replace("### ", "")
        .replace("#", "")
        .replace("> ", "")
        .replace("~~", "")
    )


def fix_incorrect_year(refs: List[Path], response: str, max_fix_range: int = 60) -> WrittenChap:
    """
    修正引用中的年份
    """
    year_pat = re.compile(r"(\d{4})")
    f_segs: List[FileNameSegmentation] = [FileNameSegmentation(p) for p in refs]  # File name segmentation
    for seg in f_segs:
        if len(seg.authors_first_segment) <= 3:
            logger.debug(f"{seg.authors}|作者名字太短,检测适用效果差，跳过")
            continue

        start_index = 0
        while True:

            author_index = response.find(seg.authors_first_segment, start_index)

            if author_index == -1:
                break
            year_match = year_pat.search(response, author_index)
            if year_match is None:
                break
            year_start, year_end = year_match.span(1)
            distance = year_start - author_index
            if distance > max_fix_range:
                logger.debug(f"{seg.authors}|距离太远: {distance}")
                break
            if response[year_start:year_end] != seg.year:
                logger.info(f"{seg.authors}|修正年份:  {response[year_start:year_end]} -> {seg.year}")
                response = response[:year_start] + seg.year + response[year_end:]
            start_index = year_end

    return response


class FileNameSegmentation:
    """
    用于解析文件名的类
    """

    def __init__(self, f_path: Path):
        self._f_name = f_path.stem
        segs = self._f_name.split(" - ")
        self._authors = segs[0].replace("et al.", "等人")
        self._year = segs[1]
        self._title = segs[2]

        self._source = f_path

    @property
    def source(self):
        """
        :return: 文件路径
        """
        return self._source

    @property
    def f_name(self):
        """
        :return: 文件名
        """
        return self._f_name

    @property
    def authors(self):
        """
        :return: 作者
        """
        return self._authors

    @property
    def authors_first_segment(self):
        """
        :return: 作者的姓
        """
        return self._authors.split(", ")[0].split(" ")[0].strip()

    @property
    def year(self):
        """
        :return: 年份
        """
        return self._year

    @property
    def title(self):
        """
        :return: 标题
        """
        return self._title


def generate_word_variants(word: str) -> List[str]:
    """
    Generates different variants of a word.
    Args:
        word: The word to generate variants for.

    Returns:
        A list of different variants of the word.
    """

    # 创建不同大小写形式的列表
    word = word.lower()
    capitalize = word.capitalize()
    upper = word.upper()
    variants = [f"{capitalize}\n", f"{upper}\n", f"{capitalize} \n", f"{upper} \n", capitalize, upper, word]
    return variants


def remove_references(article_text: str) -> str:
    """
    Removes the references section from the text.
    Args:
        article_text: The text to be processed.

    Returns:
        The text with the references section removed.

    """
    for atp in generate_word_variants("References") + generate_word_variants("参考文献"):
        segs = article_text.split(atp)
        if len(segs) > 1:
            segs.pop()
            return atp.join(segs)

    logger.warning("未找到参考文献部分")
    # If no references section is found, return the original text
    return article_text


def split_head_and_body(text: str) -> Tuple[str, str]:
    """
    Splits the text into the head and body sections.
    Args:
        text: The text to be split.

    Returns:
        A tuple containing the head and body sections.

    """

    for atp in (
        generate_word_variants("Introduction")
        + generate_word_variants("引言")
        + generate_word_variants("Background")
        + generate_word_variants("DOI")
        + generate_word_variants("e-mail")
    ):
        segs = text.split(atp)
        if len(segs) > 1:
            return segs.pop(0), atp.join(segs)
    logger.warning("未找到文章头部")
    return "", text


def read_pdf_as_text(file_path: str):
    """
    Extracts text from a PDF file.
    Args:
        file_path: Path to the PDF file.

    Returns:

    """
    document = fitz.open(file_path)
    out = ""
    for page in document.pages():
        out += page.get_text()  # Extract text from the page

    return out


def read_pdf_as_head_and_clean_body(file_path: str):
    """
    Extract
    """
    document = fitz.open(file_path)
    body = ""
    for page in document.pages(1):
        body += page.get_text()  # Extract text from the page
    return document[0], remove_references(body)


class PdfSummarizer:
    """
    用于从PDF文件中提取摘要的工具类
    """

    SUM_FORMAT_CONSTRAIN = """
- 文章标题: 不要翻译，保留原文
- 作者：作者名不要做任何的翻译，多个作者时要保留前三个然后后面加上"等人"
- 研究发表的年份：文献发表的年份是什么？
- 研究的背景：研究的动机和目的是什么？
- 文章的性质：【技术报告和技术说明 | 专利文献 | 会议论文 | 学位论文 | 行业标准与规范 | 案例研究/项目报告 | 政策文件和研究报告 | 书籍章节 | 评论文章 | 跨学科研究 | 其他】 的其中一种
- 核心工作：研究的主要贡献或创新点有那些？使用了什么方法或者技术？提取出最优秀的4到6点，一定要具体地说出技术或者方法的学名！
- 解决了什么问题：这项研究通过其各个核心工作都各自在什么条件下解决了什么问题？有什么应用价值？
- 与核心工作相对应的图表的题注：这项研究中的核心工作是否有对应的图表或者表格？如果有，提取它们所有的题注，不要翻译，直接用原名。没有的话就留空。
- 得出的主要结论：最后的结果和效果怎么样？是否达到了预期的目标？可以如何改进？一定要指出具体的指标！
- 文章的关键词：文章中研究内容所围绕的关键词是什么？给出最重要的5到8个关键词,不要翻译。
- 研究最终会影响到的对象：这项研究对学术界或者工业界的那些对象有什么影响，是否有什么进一步的应用或者研究方向？
- 研究达到的效果或影响：最终所达到的效果或者影响具体是什么？
- 研究的局限性：这项研究的局限性是什么？是否有什么可以改进的地方？
- 最终概述：将上面提取出来的内容全部按照规范的格式组织起来（
规范(注意:【】和其中的文字共同组成并指代了一个提取到的内容)：
【作者1的姓】, 【作者2的姓】, 【作者3的姓】等人，在【发表年份】里【研究背景】下的【文章的性质】中讨论了【核心工作1】在处理【解决的问题1】，此外其还讨论了【核心工作2】在处理【解决的问题2】，最终得到了【主要结论】的结论)，对【影响对象】有着【影响或者效果】.
）
请确保你的回答简洁明了，并按照以下格式组织信息，下面是一个模板用作参考（注意：最后你给出的提取内容不应该带【】）：

文章标题：【文章标题】
作者：【作者1】；【作者2】；【作者3】
发表年份：【发表年份】
研究背景：【研究背景】
文章性质：【文章性质】
核心工作：【核心工作1】；【核心工作2】；...
解决什么问题：【解决问题1】；【解决问题2】；...
与核心工作相对应的图表的题注：【图标题注1】；【图标题注2】；...
得出的主要结论：【得出的主要结论】
文章关键词：【关键词1】；【关键词2】；...
研究最终会影响到的对象：【影响对象】
研究达到的效果或影响：【效果或影响】
研究的局限性：【局限性】
最终概述：【最终概述】

下面是一个实际例子：
文章标题：Enhancing Wind Turbine Blade Efficiency through Advanced Aerodynamic Design and Material Optimization
作者：Smith；Johnson；Williams等人
发表年份：2023
研究背景：随着全球对可再生能源的需求增长，提高风力发电效率成为重要课题。本研究旨在通过先进的空气动力学设计和材料优化来增强小微风电叶片的性能。
文章性质：会议论文
核心工作：采用计算流体力学(CFD)进行叶片气动外形优化；利用有限元分析(FEA)评估结构完整性；实施多学科设计优化(MDO)以提升综合性能；开发新型复合材料提高耐久性。
解决什么问题：CFD优化解决了传统设计方法无法充分考虑复杂流动条件的问题；FEA确保了在极端负载条件下叶片的可靠性和安全性；MDO整合了多个工程学科的知识，解决了单一学科设计的局限性；新型材料的应用提高了叶片的耐用性和抗疲劳性能。
与核心工作相对应的图表的题注：Figure 1: CFD Simulation Results for Blade Aerodynamic Performance；Figure 2: FEA Stress Analysis of Optimized Blade Structure；Table 1: Comparison of Conventional and Advanced Composite Materials；Figure 3: MDO Workflow Diagram
得出的主要结论：研究结果表明，通过上述技术手段，可以将小微风电叶片的能量转换效率提高15%，同时降低了约20%的制造成本。这些改进使得小微风电系统更加经济可行，符合预期目标。
文章关键词：Wind turbine blade；Aerodynamic design；Material optimization；Computational fluid dynamics；Finite element analysis
研究最终会影响到的对象：学术界的小型风电研究者；工业界的风电设备制造商；政策制定者以及环保组织。
研究达到的效果或影响：提高了小微风电系统的市场竞争力，并促进了相关政策支持和技术标准的发展。
研究的局限性：研究主要基于模拟数据，实际测试可能因环境变量而有所差异；新材料的大规模生产可能会遇到技术和经济上的挑战。
最终概述：Smith, Johnson, Williams等人，在2023年里随着全球对可再生能源需求的增长背景下，通过会议论文讨论了采用计算流体力学(CFD)进行叶片气动外形优化、利用有限元分析(FEA)评估结构完整性、实施多学科设计优化(MDO)以提升综合性能、开发新型复合材料提高耐久性的方法在处理提高小微风电叶片能量转换效率和降低成本方面的问题，最终得到了显著提高小微风电叶片性能并降低制造成本的结论，对学术界的小型风电研究者、工业界的风电设备制造商、政策制定者以及环保组织有着促进相关政策支持和技术标准发展的影响。
"""

    def __init__(self, pdf_path: Path):
        self._seg = FileNameSegmentation(pdf_path)
        self._pdf_path = pdf_path
        self._head, self._body = read_pdf_as_head_and_clean_body(pdf_path.as_posix())

    def summary_instruction(self, max_briefing_len: int = 3800) -> str:
        """
        生成摘要提取任务指令
        """
        return (
            f"# 论文主体\n"
            f"{self._body}\n"
            f"# 论文头部\n"
            f"{self._head}\n"
            f"根据上面这篇由{self._seg.authors}等人在{self._seg.year}年发表的标题为‘{self._seg.title}’的论文，"
            f"为我完成内容提取，具体的格式要求如下：{self.SUM_FORMAT_CONSTRAIN}\n"
            f"最后的字数控制在{max_briefing_len}以内。"
        )

    @property
    def source(self) -> Path:
        """
        获取PDF文件路径
        """
        return self._pdf_path
