from pathlib import Path
from typing import List

from loguru import logger

from crazy_functions.articles.article_utils import (
    remove_markdown_syntax,
    ContentPacker,
    FileNameSegmentation,
    read_pdf_as_text,
    remove_references,
    split_head_and_body,
    PdfSummarizer,
)
from crazy_functions.crazy_utils import input_clipping
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import CatchException, report_exception
from toolbox import promote_file_to_downloadzone
from toolbox import update_ui


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
        logger.info(info := f"[{f_no+1}/{all_file_count}]begin analysis on: {file_path.as_posix()}")
        ############################## <第 0 步，切割PDF> ##################################
        # 递归地切割PDF文件，每一块（尽量是完整的一个section，比如introduction，experiment等，必要时再进行切割）
        # 的长度必须小于 2500 个 Token
        trunk = remove_references(read_pdf_as_text(file_path.as_posix()))  # （尝试）按照章节切割PDF
        head, body = split_head_and_body(trunk)

        from crazy_functions.pdf_fns.breakdown_txt import (
            breakdown_text_to_satisfy_token_limit,
        )

        paper_fragments = breakdown_text_to_satisfy_token_limit(
            txt=body,
            limit=token_limit_per_fragment,
            llm_model=llm_kwargs["llm_model"],
        )
        # 为了更好的效果，我们剥离Introduction之后的部分（如果有）

        ############################## <第 1 步，从摘要中提取高价值信息，放到history中> ##################################
        final_results = [head]

        ############################## <第 2 步，迭代地历遍整个文章，提取精炼信息> ##################################

        chatbot.append([info, ">>>"])
        yield from update_ui(chatbot=chatbot, history=[])  # 更新UI

        iteration_results = []
        last_iteration_result = head  # 初始值是摘要

        n_fragment = len(paper_fragments)

        if n_fragment > 1:
            logger.info(f"the paper is long, divided into {n_fragment} fragments. Start info pre-extraction.")
            yield from seg_sum(
                chatbot,
                iteration_results,
                last_iteration_result,
                llm_kwargs,
                max_word_total,
                n_fragment,
                paper_fragments,
            )
        else:
            logger.info(f"the paper is short, only one fragment. Skip info pre-extraction.")
            iteration_results.append(paper_fragments[0])
        ############################## <第 3 步，整理history，提取总结> ##################################
        final_results.extend(iteration_results)
        final_results.append(f"")

        f_segs = FileNameSegmentation(file_path)

        field_string = f'我正在撰写一篇关于"{title}"的文献综述论文。' if title else ""
        i_say = f"""
根据上面这篇由{f_segs.authors}在{f_segs.year}年发表的标题为“{f_segs.title}”的论文，为我完成内容提取，如下是对于一篇论文所对应综述内容的格式要求：
{format_constrain}
        """
        if additional_constrain:
            i_say += "额外的：\n"
            i_say += additional_constrain
        i_say, final_results = input_clipping(i_say, final_results, max_token_limit=max_word_total)
        gpt_say: str = yield from request_gpt_model_in_new_thread_with_ui_alive(
            inputs=i_say,
            inputs_show_user="开始最终总结",
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            history=final_results,
            sys_prompt=f"According to the given format requirements, complete the content extraction. Please ensure that the extracted content does not exceed {max_briefing_len} words",
        )

        final_results.append(remove_markdown_syntax(gpt_say))

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
                default_value="小微风电叶片仿真分析及性能优化研究",
                type="string",
            ).model_dump_json(),
            "token_restrains": ArgProperty(
                title="token restrains",
                description="The token restrains, in order, max api token|max briefing token|max segment token",
                default_value=f"{int(80000)}|{3800}|{int(80000)}",
                type="string",
            ).model_dump_json(),
            "format_constraint": ArgProperty(
                title="format constraint",
                description="The format constraint",
                default_value=PdfSummarizer.SUM_FORMAT_CONSTRAIN,
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


def seg_sum(chatbot, iteration_results, last_iteration_result, llm_kwargs, max_word_total, n_fragment, paper_fragments):
    for i in range(n_fragment):
        NUM_OF_WORD = max_word_total // n_fragment
        i_say = (
            f"Read this section, recapitulate the content of this section with less than {NUM_OF_WORD} Chinese characters: {paper_fragments[i]},"
            f" DO NOT transcribe the equations or formulas directly, but describe their mechanism and function in Chinese."
            f" DO remember mention the charts' or tables' serial numbers and names in the way the were in the original text, they are high value info, you should add them all to the final result"
            f" DO NOT try to recapitulate the references, they are not the main content of the paper, so they are useless here."
            f" DO NOTICE the name of entity should always be annotated with academic name, like 'Deep Learning' instead of 'DL'."
        )
        i_say_show_user = f"[{i + 1}/{n_fragment}] Read this section, recapitulate the content of this section with less than {NUM_OF_WORD} Chinese characters: {paper_fragments[i][:200]}"
        gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
            i_say,
            i_say_show_user,  # i_say=真正给chatgpt的提问， i_say_show_user=给用户看的提问
            llm_kwargs,
            chatbot,
            history=[
                "The main idea of the previous section is?",
                last_iteration_result,
            ],  # 迭代上一次的结果
            sys_prompt="Extract the main idea of this section with Chinese.YOU SHALL NOT Translate author's name",  # 提示
        )
        iteration_results.append(gpt_say)
        last_iteration_result = gpt_say
