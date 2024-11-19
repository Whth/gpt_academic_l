import zipfile
from pathlib import Path

from crazy_functions.make_articles import ArticleMaker
from crazy_functions.make_pdf_briefing import BriefingMaker
from toolbox import CatchException


@CatchException
def unzip_file(zip_file_path: Path, output_dir: Path) -> Path:
    """
    解压指定的 ZIP 文件到指定的输出目录。

    :param zip_file_path: ZIP 文件的路径
    :param output_dir: 输出目录的路径
    :return: 解压后的文件路径
    """
    # 确保输出目录存在
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # 解压文件
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # 返回解压后的目录路径
    return output_dir


class MakeArticleEnhance(ArticleMaker, BriefingMaker):

    def define_arg_selection_menu(self):
        gui_def = {}
        gui_def.update(ArticleMaker.define_arg_selection_menu(self))
        gui_def.update(BriefingMaker.define_arg_selection_menu(self))

        return gui_def

    def execute(
        txt: str,
        llm_kwargs,
        plugin_kwargs,
        chatbot,
        history,
        system_prompt,
        user_request,
    ):

        root = Path(txt)
        # returns a path to a zip file
        out_path = yield from BriefingMaker.execute(
            txt,
            llm_kwargs,
            plugin_kwargs,
            chatbot,
            history,
            system_prompt,
            user_request,
        )

        unziped_path = unzip_file(out := Path(out_path), root / f"{out.name}.unzip").as_posix()
        plugin_kwargs["main_input"] = unziped_path
        yield from ArticleMaker.execute(
            unziped_path,
            llm_kwargs,
            plugin_kwargs,
            chatbot,
            history,
            system_prompt,
            user_request,
        )
