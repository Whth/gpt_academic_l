import json
import tempfile
import zipfile
from pathlib import Path
from typing import List, TypeAlias, Dict
from typing import Self

from loguru import logger

from crazy_functions.crazy_utils import (
    request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
)
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import report_exception, promote_file_to_downloadzone
from toolbox import update_ui

Briefing: TypeAlias = str


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
    def relation_asm(self, briefing: Briefing) -> str:
        """
        生成用于判断文献综述与提纲关系的 ASM 任务
        """
        return (
            f"现在有一篇这个论文的综述，如下：\n"
            f"{briefing}\n\n"
            f"现在说明任务，我在编写题目为“{self.TITLE}”的论文，这个是我的论文里面的部分提纲:\n\n"
            f"{self.content}\n\n"
            f"你觉得先前给出的那个论文综述是否和我上面给出章节提纲具有相关性？以\"是否存在可迁移到我的文章的理论或者内容/方法\"作为一个判断标准,\n"
            f"额外的，还可以考虑这篇论文中是否可以将其中的一些比较有代表性的图表在我的综述中作为引用说明，这也可以作为判断标准。\n"
            f"如果是肯定的话，也就是说有关系，这个论文可以被用来作为我的论文的论述辅助材料的，就只回复关键字“{self.AFFIRMATIVE}”\n"
            f"如果是否定的话，也就是说没有关系，这个论文基本不能很好的插入到上面提纲部分内作为论述辅助材料，就只回复关键字“{self.REJECT}”\n"
            f"除了关键字外不要有额外的说明！你只需要回复“{self.AFFIRMATIVE}”或者“{self.REJECT}”两个关键字中的其中一个，你的回复中如果存在其他的任何解释都会被视为非法输入。"
        )

    def write_batch_asm(self) -> str:
        """
        生成用于撰写文章内容的 ASM 任务
        """
        ref_materials = self.load_self_references()
        asm_ref_material: str = "\n\n".join([f"[{i}]: {ref_material}" for i, ref_material in enumerate(ref_materials)])
        return f"""
{asm_ref_material}       

上面这些是我对一些相关文献总结出来的总共{len(ref_materials)}篇文献综述.
现在开始说明任务，我在编写题目为“{self.TITLE}”的论文，这个是我的论文里面的部分提纲你要专注于融合上面{len(ref_materials)}篇的文献综述补全这个提纲，变为一个完整的章节:

{self.content}

给出的所有的{len(ref_materials)}篇文献均需要被作为引文融入到我的这一个章节内，不允许有遗漏！
请你根据上面的提纲，结合上面的文献综述，为我完成这个章节的撰写，注意不要直接复制粘贴，要进行适当的融合和改写，使得这个章节的内容更加丰富和完整.
并且，一定要注意在正文中引用时要带上前三个作者的姓与年份，这里假设作者A，B，C是虚拟的的作者，正常情况下你需要使用实际对应的作者名来替换它们,一般如果一篇论文有作者三个以上，你可以说“作者A，作者B，作者C等人（2013）做了什么什么,...”。如果作者只有两个，你可以说“作者A与作者B（2013）做了什么什么,...”。如果只有一个人，你可以说“作者A（2013）做了什么什么,...”。
同样的，如果你需要同时引用多篇论文，那么你就要在年份后面加入多个占位标签字段，比如“Sam（2019），Bob（2024）做了什么什么，...”。
额外的，你可以参考参考文献中的图表，如果图表中存在可以迁移到我的文章中的图表来作为对于特定内容的辅助说明，你可以在正文中引用这些图表，不要忘了在引用后面加入占位标签字段。引文直接在正文中使用图或表的名称，比如你要引用作者A，作者B，作者C等人（2016）的论文中的“图-叶片疲劳曲线”，你可以说“作者A，作者B，作者C等人（2016）的图-叶片疲劳曲线所示，表明了什么什么，印证了什么什么，...”。
章节编号一定要按照我的提纲的来，不要自己随意增加或者减少章节。最后的你给出结果的末尾你也不用添加参考文献的尾注，我会自行添加。
也不要使用#或者*，你应该严格按照x x.y x.y.z这样的标题序号规范排版。      
除了答案外不要有额外的说明！你不用在开始写的时候表示“好的”，也不用在写完了之后表示“完成了”。

"""



    def _write_iter_asm_inner(self,grouped:List[Briefing],written_article_place_holder:str)->str:
        """
        生成用于增量撰写文章内容的 ASM 任务
        """
        asm_ref_material: str = "\n\n".join([f"[{i}]: {ref_material}" for i, ref_material in enumerate(grouped)])
        return (
            f"{asm_ref_material}"
            f"现在开始说明任务，我在编写题目为“{self.TITLE}”的论文，这个是我的论文里面的部分提纲：\n\n"
            f"{self.content}\n\n"
            f"现在我完成了如下的一部分的编写：\n\n"
            f"{written_article_place_holder}\n\n"
            f"为我完成这个章节的剩余部分的撰写，注意不要直接复制粘贴，注意一定要使得行文流程，具有逻辑，使得这个章节的内容更加丰富和完整.\n"
            "并且，一定要注意在正文中引用时要带上前三个作者的姓与年份，这里假设作者A，B，C是虚拟的的作者，正常情况下你需要使用实际对应的作者名来替换它们,"
            "一般如果一篇论文有作者三个以上，你可以说“作者A，作者B，作者C等人（2013）做了什么什么,...”。"
            "如果作者只有两个，你可以说“作者A与作者B（2013）做了什么什么,...”。如果只有一个人，你可以说“作者A（2013）做了什么什么,...”。\n"
            "同样的，如果你需要同时引用多篇论文，那么你就要在年份后面加入多个占位标签字段，比如“Sam（2019），Bob（2024）做了什么什么，...”。\n"
            "额外的，你可以参考参考文献中的图表，如果图表中存在可以迁移到我的文章中的图表来作为对于特定内容的辅助说明，"
            "你可以在正文中引用这些图表，不要忘了在引用后面加入占位标签字段。引文直接在正文中使用图或表的名称，比如你要引用作者A，作者B，作者C等人（2016）的论文中的“图-叶片疲劳曲线”，"
            "你可以说“如作者A，作者B，作者C等人（2016）的图-叶片疲劳曲线所示，表明了什么什么，印证了什么什么，...”。\n"
            "额外的，你不用在开始写的时候表示“好的”，也不用在写完了之后表示“完成了”,直接给出结果就可以。\n"
            "除了答案外不要有额外的说明！也不要使用#或者*，你应该严格按照x x.y x.y.z这样的标题序号规范排版。"      
            f"请你根据上面的提纲和我已经完成的部分内容，将上面的{len(grouped)}篇文献综述全部增量插入到我的已完成的部分章节里面，不允许有遗漏，插入的时候不应该损坏原有的文献引用，"
            "章节编号一定要按照我的提纲的来，不准自己随意增加或者减少章节。最后的你给出结果的末尾你也不用添加参考文献的尾注，我会自行添加。"
        )

    def write_iter_asm(self,per_iter_size:int=4,written_article_place_holder:str="__written__") -> List[str]:
        """
        生成用于增量撰写文章内容的 ASM 任务
        """
        ref_materials:List[Briefing] = self.load_self_references()
        grouped_ref_materials = [ref_materials[i:i + per_iter_size] for i in range(0, len(ref_materials), per_iter_size)]
        return [self._write_iter_asm_inner(grouped,written_article_place_holder) for grouped in grouped_ref_materials]

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

    @classmethod
    def check_pass(cls, refs: List[Path], response: List[str]) -> List[Path]:
        """
        检查文献综述与提纲关系的 ASM 任务的回答是否符合要求
        """
        assert len(refs) == len(
            response
        ), f"The length of refs and response should be the same, ref: {len(refs)}, response: {len(response)}"
        return list(
            map(lambda a:a[0],filter(
                lambda x: x[1] == cls.AFFIRMATIVE or cls.AFFIRMATIVE in x[1],
                zip(refs, response),
            ))
        )


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
class ArticleMaker(GptAcademicPluginTemplate):
    """
    用于生成文章内容的插件
    """

    def define_arg_selection_menu(self):

        gui_definition = {
            "main_input": ArgProperty(
                title="doc pack path",
                description="未指定路径，请上传文件后，再点击该插件",
                default_value="",
                type="string",
            ).model_dump_json(),  # 主输入，自动从输入框同步
            "outline": ArgProperty(
                title="outline",
                description="the outline of the article,note that each chapter needs to be separated by '\\n\\n', DO NOT add chapter like '参考文献'",
                default_value="""
3 仿真分析
3.1 气动性能仿真
3.1.1 不同风速下的流场模拟
3.1.2 升力、阻力、扭矩计算
3.1.3 气动效率评估
3.1.4 风能转换效率分析
3.2 结构强度仿真
3.2.1 应力分布分析
3.2.2 振动特性分析
3.2.3 耐久性评估

4 性能优化
4.1 几何形状优化
4.2 材料选择优化
4.3 截面翼型优化
4.4 多目标优化策略
4.5 优化结果与讨论

5 实验验证
5.1 实验设计
5.2 实验实施
5.3 实验结果与仿真结果对比
5.4 结果讨论""",
                type="string",
            ).model_dump_json(),
            "title": ArgProperty(
                title="title",
                description="the title of the article, as it can make the relation more accurate and confined",
                default_value="",
                type="string",
            ).model_dump_json(),
            "max_judges_threads": ArgProperty(
                title="max_judges_threads",
                description="the max number of threads to use for judging",
                default_value=int(2).__str__(),
                type="string",
            ).model_dump_json(),
            "max_write_threads": ArgProperty(
                title="max_write_threads",
                description="the max number of threads to use for writing",
                default_value=int(1).__str__(),
                type="string",
            ).model_dump_json(),
        }
        return gui_definition

    def execute(
        txt: str,
        llm_kwargs,
        plugin_kwargs,
        chatbot,
        history,
        system_prompt,
        user_request,
    ):
        chatbot.append(
            [
                "函数插件功能？",
                "这个插件的功能是将给定的文献综述和提纲，生成一个完整的文章内容。",
            ]
        )

        root = Path(txt)
        if not root.exists():
            report_exception(
                chatbot,
                history,
                a=f"解析项目: {txt}",
                b=f"找不到本地项目或无权访问: {txt}",
            )
            yield from update_ui(chatbot=chatbot, history=history)
        yield from update_ui(chatbot=chatbot, history=[])
        ref_paths: List[Path] = list(root.rglob("*.txt"))
        jsons:List[Path]=list(root.rglob("*citation_info.json"))
        pre_defined_reference={}
        if jsons:
            pre_defined_reference = json.loads(jsons.pop(0).read_text("utf-8"))

        chapters = plugin_kwargs["outline"].split("\n\n")
        title = plugin_kwargs["title"]
        ChapterOutline.TITLE = title
        max_judges_threads = int(plugin_kwargs["max_judges_threads"])
        ChapterOutline.MAX_WORKER = max_judges_threads
        max_write_threads = int(plugin_kwargs["max_write_threads"])
        chap_outlines = [ChapterOutline(content, llm_kwargs, chatbot) for content in chapters]

        for parg in chap_outlines:
            yield from parg.update_related_references(ref_paths,pre_defined_reference)
            logger.info(f"已经处理完{parg.chap_header}的文献综述, 使用了{len(parg.references)}篇文献")
        dump_materials(chap_outlines, chatbot, root)
        dump_ref_usage_manifest(chap_outlines, ref_paths, chatbot)
        gpt_res:List[str] = yield from write_article(chap_outlines, chatbot, llm_kwargs, max_write_threads)
        out_path = dump_final_result(chap_outlines, chatbot, gpt_res, root)
        yield from update_ui(chatbot=chatbot, history=history)
        return out_path



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
