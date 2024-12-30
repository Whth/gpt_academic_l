from pathlib import Path
from typing import List

from crazy_functions.articles.article_utils import ChapterOutline, dump_final_result
from crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import report_exception
from toolbox import update_ui


class ArticleMakerLite(GptAcademicPluginTemplate):
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


        chapters = plugin_kwargs["outline"].split("\n\n")
        title = plugin_kwargs["title"]
        ChapterOutline.TITLE = title
        max_write_threads = int(plugin_kwargs["max_write_threads"])
        chap_outlines = [ChapterOutline(content, llm_kwargs, chatbot) for content in chapters]

        generation_materials:List[str]=[]
        for chap in chap_outlines:
            target:Path = list(filter(lambda x: chap.chap_header in x.stem, ref_paths)).pop()
            generation_materials.extend(target.read_text("utf-8"))

        collections = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            handle_token_exceed=False,
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            inputs_array=generation_materials,
            inputs_show_user_array=[f"Dealing with {chap.chap_header}" for chap in chap_outlines],
            history_array=[[]] * len(chap_outlines),
            sys_prompt_array=[f'You are expert about how to write a paper about "{ChapterOutline.TITLE}" with given references']
                             * len(chap_outlines),
            max_workers=max_write_threads,
        )
        gpt_res=collections[1::2]
        out_path=dump_final_result(gpt_res, chap_outlines, chatbot,root)
        yield from update_ui(chatbot=chatbot, history=history)
        return out_path


