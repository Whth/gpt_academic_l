import tempfile
import zipfile
from pathlib import Path
from typing import List
from typing import Self

from loguru import logger

from crazy_functions.crazy_utils import input_clipping
from crazy_functions.crazy_utils import read_and_clean_pdf_text
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import CatchException, report_exception
from toolbox import promote_file_to_downloadzone
from toolbox import update_ui


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


# 使用示例


def make_brifing_inner(
    file_manifest: List[Path],
    llm_kwargs,
    chatbot,
    token_limit_per_fragment: int,
    max_word_total: int,
    title: str,
    max_briefing_len: int,
    format_constrain: str,
    additional_constrain: str,
):
    packer = ContentPacker()
    all_file_count = len(file_manifest)
    out_path = ""
    for f_no, file_path in enumerate(file_manifest):
        file_name = file_path.as_posix()
        logger.info(info := f"begin analysis on: {file_name}")
        ############################## <第 0 步，切割PDF> ##################################
        # 递归地切割PDF文件，每一块（尽量是完整的一个section，比如introduction，experiment等，必要时再进行切割）
        # 的长度必须小于 2500 个 Token
        file_content, page_one = read_and_clean_pdf_text(file_name)  # （尝试）按照章节切割PDF
        file_content = file_content.encode("utf-8", "ignore").decode()  # avoid reading non-utf8 chars
        page_one = str(page_one).encode("utf-8", "ignore").decode()  # avoid reading non-utf8 chars

        from crazy_functions.pdf_fns.breakdown_txt import (
            breakdown_text_to_satisfy_token_limit,
        )

        paper_fragments = breakdown_text_to_satisfy_token_limit(
            txt=file_content,
            limit=token_limit_per_fragment,
            llm_model=llm_kwargs["llm_model"],
        )
        page_one_fragments = breakdown_text_to_satisfy_token_limit(
            txt=str(page_one),
            limit=token_limit_per_fragment // 4,
            llm_model=llm_kwargs["llm_model"],
        )
        # 为了更好的效果，我们剥离Introduction之后的部分（如果有）
        paper_meta = page_one_fragments[0].split("introduction")[0].split("Introduction")[0].split("INTRODUCTION")[0]

        ############################## <第 1 步，从摘要中提取高价值信息，放到history中> ##################################
        final_results = [paper_meta]

        ############################## <第 2 步，迭代地历遍整个文章，提取精炼信息> ##################################

        chatbot.append([info, ">>>"])
        yield from update_ui(chatbot=chatbot, history=[])  # 更新UI

        iteration_results = []
        last_iteration_result = paper_meta  # 初始值是摘要

        n_fragment = len(paper_fragments)
        if n_fragment >= 20:
            logger.warning("文章极长，可能无法达到预期效果")
        for i in range(n_fragment):
            NUM_OF_WORD = max_word_total // n_fragment
            i_say = f"Read this section, recapitulate the content of this section with less than {NUM_OF_WORD} Chinese characters: {paper_fragments[i]}"
            i_say_show_user = f"[{i+1}/{n_fragment}] Read this section, recapitulate the content of this section with less than {NUM_OF_WORD} Chinese characters: {paper_fragments[i][:200]}"
            gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
                i_say,
                i_say_show_user,  # i_say=真正给chatgpt的提问， i_say_show_user=给用户看的提问
                llm_kwargs,
                chatbot,
                history=[
                    "The main idea of the previous section is?",
                    last_iteration_result,
                ],  # 迭代上一次的结果
                sys_prompt="Extract the main idea of this section with Chinese.YOU SHALL NOT Translate author's name into chinese",  # 提示
            )
            iteration_results.append(gpt_say)
            last_iteration_result = gpt_say

        ############################## <第 3 步，整理history，提取总结> ##################################
        final_results.extend(iteration_results)
        final_results.append(f"")

        field_string = f'我正在撰写一篇关于"{title}"的文献综述论文。' if title else ""
        i_say = f"""
根据上面这这篇论文，为我完成内容提取，具体的格式要求如下：
{field_string}这个是对于一篇论文所对应综述内容的格式要求：

{format_constrain}

        """
        if additional_constrain:
            i_say += "额外的：\n"
            i_say += additional_constrain
        i_say, final_results = input_clipping(i_say, final_results, max_token_limit=max_word_total)
        gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=i_say,
            inputs_show_user="开始最终总结",
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            history=final_results,
            sys_prompt=f"According to the given format requirements, complete the content extraction. Please ensure that the extracted content does not exceed {max_briefing_len} words",
        )
        final_results.append(gpt_say)

        ############################## <第 4 步，设置一个token上限> ##################################
        _, final_results = input_clipping("", final_results, max_token_limit=max_word_total)

        (
            packer.add_content(file_path.stem, str(gpt_say)).pack(
                out := f"{file_path.parent}/{f_no+1}-{all_file_count}.zip"
            )
        )
        logger.info("zip file created:", out)
        out_path = promote_file_to_downloadzone(out, chatbot=chatbot, rename_file=Path(out).name)
        yield from update_ui(chatbot=chatbot, history=final_results)  # 注意这里的历史记录被替代了

    packer.cleanup()
    return out_path


@CatchException
def make_brifing(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    import os

    # 基本信息：功能、贡献者
    chatbot.append(
        [
            "函数插件功能？",
            "批量提取PDF文献综述。author：Whth, derived from ValeriaWong，Eralien's",
        ]
    )
    yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        import fitz
    except:
        report_exception(
            chatbot,
            history,
            a=f"解析项目: {txt}",
            b=f"导入软件依赖失败。使用该模块需要额外依赖，安装方法```pip install --upgrade pymupdf```。",
        )
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return

    # 清空历史，以免输入溢出
    history = []

    # 检测输入参数，如没有给定输入参数，直接退出
    if os.path.exists(txt):
        project_folder = txt
    else:
        if txt == "":
            txt = "空空如也的输入栏"
        report_exception(chatbot, history, a=f"解析项目: {txt}", b=f"找不到本地项目或无权访问: {txt}")
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return

    # 搜索需要处理的文件清单
    # 使用 rglob 方法递归搜索所有 .pdf 文件
    file_manifest: List[Path] = list(Path(project_folder).rglob("*.pdf"))

    # 如果没找到任何文件
    if len(file_manifest) == 0:
        report_exception(chatbot, history, a=f"解析项目: {txt}", b=f"找不到任何.pdf文件: {txt}")
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return
    # 解析 token_restrains 参数
    # description="The token restrains, in order, max api token|max briefing token|max segment token",

    token_restrains = tuple(map(int, plugin_kwargs["token_restrains"].split("|")))
    max_api_token, max_briefing_len, max_segment_token = token_restrains

    # 其他参数直接从 plugin_kwargs 中获取
    title = plugin_kwargs["title"]
    format_constrain = plugin_kwargs["format_constraint"]
    additional_constrain = plugin_kwargs["additional_constraint"]

    return (
        yield from make_brifing_inner(
            file_manifest,
            llm_kwargs,
            chatbot,
            token_limit_per_fragment=max_segment_token,
            max_briefing_len=max_briefing_len,
            title=title,
            max_word_total=max_api_token,
            format_constrain=format_constrain,
            additional_constrain=additional_constrain,
        )
    )


class BriefingMaker(GptAcademicPluginTemplate):

    def define_arg_selection_menu(self):

        std_format = """
- 文章标题: 不要翻译，保留原文
- 作者：作者名不要做任何的翻译，多个作者时要保留前三个然后后面加上"等"
- 研究发表的年份：文献发表的年份是什么？
- 研究的背景：研究的动机和目的是什么？
- 文章的性质：【技术报告和技术说明 | 专利文献 | 会议论文 | 学位论文 | 行业标准与规范 | 案例研究/项目报告 | 政策文件和研究报告 | 书籍章节 | 评论文章 | 跨学科研究 | 其他】 的其中一种
- 核心工作：研究的主要贡献或创新点，使用了什么方法或者技术，提出1或者2点，一定要具体地说出技术或者方法的学名！
- 解决了什么问题：这项研究通过核心工作解决了什么问题，有什么应用价值，提出1或者2点
- 得出的主要结论：最后的结果和效果怎么样，是否达到了预期的目标，可以如何改进，一定要指出具体的指标
- 研究最终会影响到的对象：这项研究对学术界或者工业界的那些对象有什么影响，是否有什么进一步的应用或者研究方向？
- 研究达到的效果或影响：最终所达到的效果或者影响具体是什么？
- 最终概述：将上面提取出来的内容全部按照规范的格式组织起来（
规范(注意:【】和其中的文字共同组成并指代了一个提取到的内容)：
【作者1的姓】, 【作者2的姓】, 【作者3的姓】等，在【发表年份】里【研究背景】下的【文章的性质】中讨论了【核心工作1】在处理【解决的问题1】，此外其还讨论了【核心工作2】在处理【解决的问题2】，最终得到了【主要结论】的结论)，对【影响对象】有着【影响或者效果】.
）
请确保你的回答简洁明了，并按照以下格式组织信息，下面是一个模板用作参考（注意：最后你给出的提取内容不应该带【】）：

文章标题：【文章标题】
作者：【作者1】；【作者2】；【作者3】
发表年份：【发表年份】
研究背景：【研究背景】
文章性质：【文章性质】
核心工作：【核心工作1】；【核心工作2】；...
解决什么问题：【解决问题1】；【解决问题2】；...
得出的主要结论：【得出的主要结论】
研究最终会影响到的对象：【影响对象】
研究达到的效果或影响：【效果或影响】
最终概述：【最终概述】
"""

        gui_definition = {
            "main_input": ArgProperty(
                title="doc pack path",
                description="未指定路径，请上传文件后，再点击该插件",
                default_value="",
                type="string",
            ).model_dump_json(),  # 主输入，自动从输入框同步
            "title": ArgProperty(
                title="research field",
                description="The research field",
                default_value="",
                type="string",
            ).model_dump_json(),
            "token_restrains": ArgProperty(
                title="token restrains",
                description="The token restrains, in order, max api token|max briefing token|max segment token",
                default_value=f"{int(4096 * 0.7)}|{2500}|{int(4096 * 0.7)}",
                type="string",
            ).model_dump_json(),
            "format_constraint": ArgProperty(
                title="format constraint",
                description="The format constraint",
                default_value=std_format,
                type="string",
            ).model_dump_json(),
            "additional_constraint": ArgProperty(
                title="additional constraint",
                description="The additional constraint",
                default_value="",
                type="string",
            ).model_dump_json(),  # 高级参数输入区，自动同步
        }
        return gui_definition

    def execute(txt: str, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):

        return (
            yield from make_brifing(
                txt,
                llm_kwargs,
                plugin_kwargs,
                chatbot,
                history,
                system_prompt,
                user_request,
            )
        )
