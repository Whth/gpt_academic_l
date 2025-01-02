import json
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from crazy_functions.crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
from crazy_functions.plugin_template.plugin_class_template import (
    GptAcademicPluginTemplate,
    ArgProperty,
)
from toolbox import report_exception
from toolbox import update_ui


class Score(BaseModel):
    references_used_count_score: float = Field(description="文章引用数量评分", ge=0, le=10)
    fluency_score: float = Field(description="文章流畅度评分", ge=0, le=10)
    coherence_score: float = Field(description="文章连贯性评分", ge=0, le=10)
    relevance_score: float = Field(description="文章相关性评分", ge=0, le=10)
    novelty_score: float = Field(description="文章新颖性评分", ge=0, le=10)
    no_duplicated_content_score: float = Field(description="文章重复内容评分", ge=0, le=10)
    chap_name: str = Field(description="章节名称", default="")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "references_used_count_score": 5.3,
                "fluency_score": 4.3,
                "coherence_score": 8.1,
                "relevance_score": 6.5,
                "novelty_score": 1.3,
                "no_duplicated_content_score": 1.9,
                "chap_name": "1. Introduction",
            }
        }
    )

    def total_score(self):
        """
        计算总评分
        """
        return sum(self.model_dump(exclude={"chap_name"}).values())

    def lowest_score_and_key(self):
        """
        计算最低分
        """
        return min(self.model_dump(exclude={"chap_name"}).items(), key=lambda x: x[1])


class ArticleScore(GptAcademicPluginTemplate):
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
        article_segs: List[Path] = list(root.rglob("*.txt"))
        place_holder = [[]] * len(article_segs)
        resp = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            handle_token_exceed=False,
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            scroller_max_len=500,
            inputs_array=[
                f"{p.read_text()}\n"
                f"阅读上面的文章段落，给出1-10的评分，评分格式的一个例子如下,仿照它给出评分：\n\n\n"
                f"{json.dumps(Score.model_json_schema()['example'],ensure_ascii=False)}\n\n\n 结果是一个纯合法的json,不要带其他任何说明,我会对你的结果直接使用json.loads"
                for p in article_segs
            ],
            inputs_show_user_array=[f"Processing {p.name}" for p in article_segs],
            history_array=place_holder,
            sys_prompt_array=place_holder,
            max_workers=len(article_segs),
            retry_times_at_unknown_error=5,
        )

        for r in resp[1::2]:

            r = r.replace("```json", "").replace("```", "")
            try:
                score = Score(**json.loads(r))

            except Exception as e:
                report_exception(
                    chatbot,
                    history,
                    a=f"解析评分: {r}",
                    b=f"解析评分失败: {e}",
                )
                continue

            logger.info(f"{score.chap_name}-评分结果: {score.total_score():2f}/60|{score.lowest_score_and_key()}")
            chatbot.append(
                [f"{score.chap_name}-评分结果", f"{score.total_score():2f}/60|{score.lowest_score_and_key()}"]
            )
            yield from update_ui(chatbot=chatbot, history=history)
