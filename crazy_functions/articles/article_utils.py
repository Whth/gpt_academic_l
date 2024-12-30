import json
import tempfile
import zipfile
from itertools import chain
from pathlib import Path
from typing import TypeAlias, List, Self, Dict, Optional

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
    TITLE=''

    def __init__(self, content: str, llm_kwargs, chatbot,references:List[Path]=None):
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

    def set_references(self, references: List[Path])->Self:
        """
        设置文献综述的路径
        """
        self.references.clear()
        self.references.extend(references)
        return self

    @staticmethod
    def load_references(ref_paths:List[Path])->List[Briefing]:
        """
        读取文献综述的内容
        """
        return [p.read_text("utf-8") for p in ref_paths]

    def load_self_references(self)->List[Briefing]:
        """
        读取文献综述的内容
        """
        return self.load_references(self.references)
    def relation_asm(self, briefing: Briefing, threshold:int=75) -> str:
        """
        生成用于评估文献综述与论文提纲相关性的 ASM 任务。
        """
        return (
            f"请根据以下标准判断这篇文献综述是否适合用作题目为“{self.TITLE}”的论文提纲辅助材料：\n\n"
            f"1. 综述中是否存在可迁移至论文提纲的理论、内容或方法。\n"
            f"2. 综述中的图表是否具有代表性，能否作为引用说明。\n\n"
    
            f"文献综述:\n{briefing}\n\n"
    
            f"论文提纲:\n{self.content}\n\n"
    
            f"如果综述与提纲高度相关（{threshold}%以上重合），回复关键字“{self.AFFIRMATIVE}”。\n"
            f"如果综述与提纲关联度低（{threshold}%以下重合），回复关键字“{self.REJECT}”。\n"
    
            f"仅回复上述两个关键字之一，任何额外解释均视为无效输入。"
        )

    def write_batch_asm(self) -> str:
        """
        生成用于撰写文章内容的ASM任务指令。
        """
        ref_materials = self.load_self_references()
        asm_ref_material = "\n\n".join([f"[{i}]: {ref}" for i, ref in enumerate(ref_materials)])

        return \
        f"""
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
            f"正在编写题目为“{self.TITLE}”的论文，根据提供的提纲和已完成的部分，将{len(grouped)}篇参考文献综述增量式插入到章节中。\n"
            "要求：\n"
            "- 不得遗漏任何文献，保持原有引用完整性。\n"
            "- 章节编号严格依照提纲，不得增减。\n"
            "- 行文流畅，逻辑严谨，丰富并完善章节内容。\n"
            "- 引用格式：根据作者数量正确使用引文格式，如“作者1，作者2，作者3等人（年份）”。\n"
            "- 可引用图表辅助说明，但仅限于字符描述其位置，不实际插入。\n"
            "- 输出为纯文本，遵循指定的标题序号规范。\n"
            "- 无需开头问候或结尾总结。\n\n"
            f"论文题目：{self.TITLE}\n\n"
            f"论文的部分提纲：\n{self.content}\n\n"
            f"已完成部分：\n{written_article_place_holder}\n\n"
            "请按照上述要求完成剩余部分的撰写。"
        )

        return instruction_template


    def write_iter_asm(self, grouped_paths:List[List[Path]], written_article_place_holder:str= "__written__") -> List[str]:
        """
        生成用于增量撰写文章内容的 ASM 任务
        """

        return [self._write_iter_asm_inner(grouped,written_article_place_holder)
                for grouped in grouped_paths]

    def content_group_by(self, group_size:int=4)->List[List[Briefing]]:
        """
        将文献综述分组
        """
        ref_materials: List[Briefing] = self.load_self_references()
        grouped_ref_materials = [ref_materials[i:i + group_size] for i in
                                 range(0, len(ref_materials), group_size)]
        return grouped_ref_materials
    def path_group_by(self, group_size:int=4)->List[List[Path]]:
        """
        将文献综述分组
        """
        grouped_ref_materials = [self.references[i:i + group_size] for i in
                                 range(0, len(self.references), group_size)]
        return grouped_ref_materials
    def update_related_references(self, briefings_path: List[Path],pre_defined_reference:Dict[str,List[str]]=None)->Self:
        """
        用于处理文献综述与提纲关系的 ASM 任务
        """
        logger.info(f"开始处理{self.chap_header}的文献综述,过滤出符合条件的文献")

        pre_defined_reference=pre_defined_reference or {}
        briefings_names=[briefing.name for briefing in briefings_path]
        if self.chap_header in pre_defined_reference and all([ref in briefings_names for ref in pre_defined_reference[self.chap_header]]):
            logger.info(f"使用预设的文献综述关系， 共{len(pre_defined_reference[self.chap_header])}篇")
            self.set_references(list(filter(lambda x:x.name in pre_defined_reference[self.chap_header],briefings_path)))
        else:
            briefings=self.load_references(briefings_path)
            res = yield from (
                request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
                    handle_token_exceed=False,
                    llm_kwargs=self._llm_kwargs,
                    chatbot=self._chatbot,
                    inputs_array=[self.relation_asm(briefing) for briefing in briefings],
                    inputs_show_user_array=[briefing.split("\n")[0] for briefing in briefings],
                    history_array=[[]] * len(briefings),
                    sys_prompt_array=[self._role] * len(briefings),
                    max_workers=self.MAX_WORKER,
                )
            )
            self.set_references(self.check_pass(briefings_path, res[1::2]))
        return self


    def write_iter_unused_asm(self,written,ref_materials:List[Path],) -> Optional[str]:
        """
        构造未使用的文献综述的 ASM 任务
        """
        unused=self.get_incited(ref_materials, written)
        if not unused:
            return None
        written_article_place_holder:str="__written__"
        return self._write_iter_asm_inner(unused,written_article_place_holder).replace(written_article_place_holder,written)
    @classmethod
    def check_pass(cls, refs: List[Path], responses: List[str]) -> List[Path]:
        """
        检查文献综述与提纲关系的 ASM 任务的回答是否符合要求
        """
        assert len(refs) == len(
            responses
        ), f"The length of refs and response should be the same, ref: {len(refs)}, response: {len(responses)}"
        return list(
            map(lambda a:a[0],filter(
                lambda x: x[1] == cls.AFFIRMATIVE or cls.AFFIRMATIVE in x[1],
                zip(refs, responses),
            ))
        )

    @staticmethod
    def get_incited(refs: List[Path], response:str) -> List[Path]:
        """
        检查文献综述与提纲关系的 ASM 任务的回答是否符合要求
        """


        f_segs:List[FileNameSegmentation]=[FileNameSegmentation(p) for p in refs] # 文件名分割
        incited_segs=filter(lambda x:x.authors.split(" ")[0] not in response,f_segs) # 未被引用的文献
        cited_segs=filter(lambda x:x.authors.split(" ")[0] in response,f_segs) # 被引用的文献
        cited_incorrectly=filter(lambda x:x.year not in response or # 作者年份不在回答中
                          abs(response.index(x.authors.split(" ")[0])-response.index(x.year))>60, # 作者年份在回答中，但是位置不对
                 cited_segs)
        return [f.source for f in chain(incited_segs,cited_incorrectly)]


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

    def cleanup(self) -> None:
        """清理临时目录"""
        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.files.clear()


class CitationMaker:
    """
    用于生成文章引用信息的工具类
    """

    def __init__(self,ref_paths:List[Path]):
        self._ref_paths = ref_paths

    def remove_used_refs(self, used_refs:List[Path])->Self:
        """
        从参考文献列表中移除已经使用过的文献
        """
        self._ref_paths = list(filter(lambda x: x not in used_refs, self._ref_paths))

        return self
    @property
    def ref_path(self)->List[Path]:
        """
        获取参考文献列表
        """
        return self._ref_paths
    def set_refs(self, refs:List[Path])->Self:
        """
        设置参考文献列表
        """
        self._ref_paths.clear()
        self._ref_paths.extend(refs)
        return self

    def remove_all_used_ref(self,chaps:List[ChapterOutline])->Self:
        """
        从参考文献列表中移除已经使用过的文献
        """
        used_refs = []
        for chap in chaps:
            used_refs.extend(chap.references)
        self.remove_used_refs(used_refs)
        return self


def dump_ref_usage_manifest(chaps: List['ChapterOutline'], all_refs: List[Path], chatbot):
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
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.json') as temp_file:
        temp_file_path = temp_file.name
        json.dump(manifest, temp_file, indent=2,ensure_ascii=False)

    # 提升文件到下载区域，并指定文件名为 "citation_info.json"
    promote_file_to_downloadzone(temp_file_path, "citation_info.json", chatbot=chatbot)


def dump_final_result( chap_outlines, chatbot, gpt_res, root):
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


def write_article( chap_outlines, chatbot, llm_kwargs, max_write_threads)->List[str]:
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
        sys_prompt_array=[f'You are expert about how to write a paper about "{ChapterOutline.TITLE}" with given references']
                         * len(chap_outlines),
        max_workers=max_write_threads,
    )
    gpt_res:List[str] = collections[1::2]
    return gpt_res


def remove_markdown_syntax(content:str)->str:
    """
    移除Markdown语法
    """
    return (content
            .replace("```","")
            .replace("```plaintext","")
            .replace("**","")
            .replace("*","")
            .replace("# ","")
            .replace("## ","")
            .replace("### ","")
            .replace("#","")
            .replace("> ","")
            .replace("~~",""))


class FileNameSegmentation:

    """
    用于解析文件名的类
    """
    def __init__(self,f_path:Path):
        self._f_name = f_path.stem
        segs=self._f_name.split(" - ")
        self._authers=segs[0].replace(" et al.","等人")
        self._year=segs[1]
        self._title=segs[2]

        self._source=f_path


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
        return self._authers
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


