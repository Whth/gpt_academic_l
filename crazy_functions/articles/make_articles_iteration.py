import json
from pathlib import Path
from time import sleep
from typing import List, TypeAlias

from loguru import logger

from crazy_functions.articles.make_articles import ChapterOutline, dump_materials, dump_final_result, \
    dump_ref_usage_manifest
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import report_exception
from toolbox import update_ui

Briefing: TypeAlias = str



class ArticleMakerIter(GptAcademicPluginTemplate):
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
            "iter_group_size": ArgProperty(
                title="iter_group_size",
                description="the number of references to be used in a iter",
                default_value=int(4).__str__(),
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
        chap_outlines = [ChapterOutline(content, llm_kwargs, chatbot) for content in chapters]

        for parg in chap_outlines:
            yield from parg.update_related_references(ref_paths,pre_defined_reference)
            logger.info(f"已经处理完{parg.chap_header}的文献综述, 使用了{len(parg.references)}篇文献")
        dump_materials(chap_outlines, chatbot, root)
        dump_ref_usage_manifest(chap_outlines, ref_paths, chatbot)
        sleep(20)
        gpt_res:List[str] = yield from write_article_iter(chap_outlines, chatbot, llm_kwargs, int(plugin_kwargs["iter_group_size"]))
        out_path = dump_final_result(chap_outlines, chatbot, gpt_res, root)


        yield from update_ui(chatbot=chatbot, history=history)
        return out_path





def write_article_iter(chap_outlines:List[ChapterOutline], chatbot, llm_kwargs,group_size:int)->List[str]:
    """
    生成文章内容
    """
    place_holder="$placebo$"
    out:List[str]=[]
    for chap in chap_outlines:
        materials:List[str]=chap.write_iter_asm(group_size,place_holder)
        logger.info(f"已经生成{chap.chap_header}的文章内容, 使用了{len(chap.references)}篇文献，迭代{len(materials)}趟")
        last_written=chap.content
        for i,material in enumerate(materials):

            last_written=yield from request_gpt_model_in_new_thread_with_ui_alive(
                inputs=material.replace(place_holder,last_written),
                inputs_show_user=f"开始第[{i+1}/{len(materials)}]轮迭代",
                history=[],
                sys_prompt="You are expert about how to insert the reference into the article appropriately without damaging the original content",
                chatbot=chatbot,
                llm_kwargs=llm_kwargs,
            )
        out.append(last_written)
    return out
