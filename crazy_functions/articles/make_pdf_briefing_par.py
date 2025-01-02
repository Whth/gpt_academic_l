from pathlib import Path
from typing import List

from crazy_functions.articles.article_utils import PdfSummarizer, ContentPacker
from crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import report_exception, promote_file_to_downloadzone
from toolbox import update_ui


# 使用示例


class BriefingMakerParallel(GptAcademicPluginTemplate):
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
            "token_restrains": ArgProperty(
                title="token restrains",
                description="the token restrains for the summarization",
                default_value=f"{3800}",
                type="string",
            ).model_dump_json(),
            "threads": ArgProperty(
                title="threads",
                description="The threads used to summarize",
                default_value="1",
                type="string",
            ).model_dump_json(),  # 高级参数输入区，自动同步
        }
        return gui_definition

    def execute(txt: str, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
        """
        插件执行函数
        """

        # 尝试导入依赖，如果缺少依赖，则给出安装建议
        try:
            import fitz
        except ModuleNotFoundError:
            report_exception(
                chatbot,
                history,
                a=f"解析项目: {txt}",
                b=f"导入软件依赖失败。使用该模块需要额外依赖，安装方法```pip install --upgrade PyMuPDF```。\n"
                f"或者使用```uv pip install --upgrade PyMuPDF```安装。",
            )
            yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
            return

        pdfs = [PdfSummarizer(p) for p in Path(txt).rglob("*.pdf")]
        if not pdfs:
            report_exception(
                chatbot,
                history,
                a=f"搜索pdf: {txt}",
                b=f"找不到任何pdf文件，请检查文件夹是否包含pdf文件。",
            )
            yield from update_ui(chatbot=chatbot, history=history)
            return
        place_holder = [[]] * len(pdfs)

        resp = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            handle_token_exceed=False,
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            scroller_max_len=500,
            inputs_array=[
                pdf.summary_instruction(max_briefing_len=int(plugin_kwargs["token_restrains"])) for pdf in pdfs
            ],
            inputs_show_user_array=[f"Processing {pdf.source.name}" for pdf in pdfs],
            history_array=place_holder,
            sys_prompt_array=place_holder,
            max_workers=int(plugin_kwargs["threads"]),
        )

        gpt_says: List[str] = resp[1::2]
        packer = ContentPacker()

        for pdf, say in zip(pdfs, gpt_says):
            packer.add_content(pdf.source.stem, say)
        promote_file_to_downloadzone(packer.pack_and_cleanup(f"{txt}/out.zip"), "briefings.zip", chatbot)
