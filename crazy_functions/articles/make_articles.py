import tempfile
import zipfile
from pathlib import Path
from typing import List, TypeAlias
from typing import Self

from loguru import logger

from crazy_functions.crazy_utils import (
    request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency,
)
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import report_exception, promote_file_to_downloadzone
from toolbox import update_ui

Briefing: TypeAlias = str


class ChapterOutline:
    AFFIRMATIVE = "是相关的"
    REJECT = "没有关系"
    MAX_WORKER = 4

    def __init__(self, title: str, content: str, llm_kwargs, chatbot):
        self.title = title
        self.content = content
        self._llm_kwargs = llm_kwargs
        self._chatbot = chatbot
        self._role = f'You are expert about how to write a paper about "{self.title}" and how to choose most appropriate references.'

    @property
    def chap_header(self) -> str:
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

    def relation_asm(self, briefing: Briefing) -> str:
        return f"""
我在编写题目为“{self.title}”的论文，这个是我的论文里面的部分提纲:

{self.content}

现在有一篇这个论文的综述，你觉得这个综述是否和我上面给出的提纲具有相关性？可以以是否存在可迁移到我的文章的理论或者内容/方法作为一个判断标准
如果是肯定的话，就只回复“{self.AFFIRMATIVE}”
如果是否定的话，就只回复“{self.REJECT}”
不要有额外的说明！     
具体所指的论文综述如下：

{briefing}

"""

    def write_asm(self, ref_materials: List[str]) -> str:
        asm_ref_material: str = "\n\n".join([f"[{i}]: {ref_material}" for i, ref_material in enumerate(ref_materials)])
        return f"""
{asm_ref_material}       

上面这些是我对一些相关文献总结出来的总共{len(asm_ref_material)}篇文献综述.
现在开始说明任务，我在编写题目为“{self.title}”的论文，这个是我的论文里面的部分提纲你要专注于融合上面{len(asm_ref_material)}篇的文献综述补全这个提纲，变为一个完整的章节:

{self.content}

请你根据上面的提纲，结合上面的文献综述，为我完成这个章节的撰写，注意不要直接复制粘贴，要进行适当的融合和改写，使得这个章节的内容更加丰富和完整。
并且，一定要注意在正文中引用时要带上前三个作者的姓与年份,一般你可以说“作者A，作者B，作者C（2013）等做了什么什么”。额外的，你不用在开始写的时候表示“好的”，也不用在写完了之后表示“完成了”。
章节编号一定要按照我的提纲的来，不要自己随意增加或者减少章节。最后的你给出结果的末尾你也不用添加参考文献的尾注，我会自行添加。
除了答案外不要有额外的说明！也不要使用#或者*，你应该严格按照x x.y x.y.z这样的标题序号规范排版，不准出现连续的两个换行符！        


"""

    def with_relation(self, briefings: List[Briefing]):
        logger.info(f"开始处理{self.chap_header}的文献综述,过滤出符合条件的文献")

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
        return res[1::2]

    @classmethod
    def check_pass(cls, refs: List[Briefing], response: List[str]) -> List[Briefing]:
        assert len(refs) == len(
            response
        ), f"The length of refs and response should be the same, ref: {len(refs)}, response: {len(response)}"
        return list(
            filter(
                lambda x: x[1] == cls.AFFIRMATIVE or cls.AFFIRMATIVE in x[1],
                zip(refs, response),
            )
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


class ArticleMaker(GptAcademicPluginTemplate):

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
                description="the outline of the article,note that each chapter needs to be separated by '\\n\\n'",
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
                default_value=int(6).__str__(),
                type="string",
            ).model_dump_json(),
            "max_write_threads": ArgProperty(
                title="max_write_threads",
                description="the max number of threads to use for writing",
                default_value=int(2).__str__(),
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

        read_contents: List[str] = [path.read_text(encoding="utf-8") for path in ref_paths]

        chapters = plugin_kwargs["outline"].split("\n\n")
        title = plugin_kwargs["title"]
        max_judges_threads = int(plugin_kwargs["max_judges_threads"])
        ChapterOutline.MAX_WORKER = max_judges_threads
        max_write_threads = int(plugin_kwargs["max_write_threads"])
        chap_outlines = [ChapterOutline(title, content, llm_kwargs, chatbot) for content in chapters]

        write_asms: List[str] = []
        for parg in chap_outlines:
            gpt_response: List[str] = yield from parg.with_relation(read_contents)
            passed_refs = parg.check_pass(read_contents, gpt_response)
            write_asms.append(parg.write_asm(passed_refs))
        packer = ContentPacker()
        for chap, w_asm in zip(chap_outlines, write_asms):
            packer.add_content(chap.chap_header, w_asm)
        packer.pack(pre_obj := (root / (f_name := f"{title}.zip")).as_posix()).cleanup()
        logger.info(f"已经生成初步原材料ZIP文件: {pre_obj}")
        promote_file_to_downloadzone(pre_obj, f_name, chatbot=chatbot)
        logger.info(f"开始生成最终的文章内容")
        collections = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            handle_token_exceed=False,
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            inputs_array=write_asms,
            inputs_show_user_array=[f"Dealing with {chap.chap_header}" for chap in chap_outlines] * len(write_asms),
            history_array=[[]] * len(write_asms),
            sys_prompt_array=[f'You are expert about how to write a paper about "{title}" with given references']
            * len(write_asms),
            max_workers=max_write_threads,
        )
        gpt_res = collections[1::2]

        for chap, resp in zip(chap_outlines, gpt_res):
            packer.add_content(f"{chap.chap_header}-write", resp)

        packer.add_content(title, "\n".join(gpt_res))
        f_path = (root / (fi_name := f"{title}-write.zip")).as_posix()
        packer.pack(f_path)
        logger.info(f"已经生成最终文章内容ZIP文件: {f_path}")
        out_path = promote_file_to_downloadzone(f_path, fi_name, chatbot=chatbot)

        yield from update_ui(chatbot=chatbot, history=history)
        return out_path
