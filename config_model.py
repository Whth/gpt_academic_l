from typing import List, Dict

from pydantic import BaseModel, AnyHttpUrl, Field, HttpUrl


class ProxyConfig(BaseModel):
    http: str = Field(default="socks5h://localhost:11284", description="HTTP代理地址，格式为 [协议]://[地址]:[端口]")
    https: str = Field(default="socks5h://localhost:11284", description="HTTPS代理地址，格式为 [协议]://[地址]:[端口]")


class AzureModelConfig(BaseModel):
    AZURE_ENDPOINT: HttpUrl = Field(..., description="The endpoint URL for the Azure model.")
    AZURE_API_KEY: str = Field(..., description="The API key for accessing the Azure model.")
    AZURE_ENGINE: str = Field(..., description="The engine name for the Azure model.")
    AZURE_MODEL_MAX_TOKEN: int = Field(..., description="The maximum number of tokens for the Azure model.")


class AzureCFGArray(BaseModel):
    azure_models: Dict[str, AzureModelConfig] = Field(
        default_factory=dict, description="A list of tuples containing Azure models configurations."
    )


class OpenAIConfig(BaseModel):
    api_key: str = Field(
        default="",
        description="OpenAI API密钥。可以同时填写多个API-KEY，用英文逗号分割，例如 'sk-openaikey1,sk-openaikey2,fkxxxx-api2dkey3,azure-apikey4'",
    )
    use_proxy: bool = Field(
        default=False,
        description="是否使用代理。如果直接在海外服务器部署或使用本地/无地域限制的大模型时，可以设置为False",
    )
    proxies: ProxyConfig = Field(default_factory=ProxyConfig, description="代理配置，包含HTTP和HTTPS的代理地址")

    embedding_model: str = Field(default="text-embedding-3-small", description="默认的嵌入模型")
    api_url_redirect: Dict[str, str] = Field(
        default_factory=dict, description="重新URL重定向配置，实现更换API_URL的作用（高危设置! 常规情况下不要修改!）"
    )
    default_worker_num: int = Field(default=3, description="多线程函数插件中，默认允许多少路线程同时访问OpenAI")
    timeout_seconds: int = Field(default=30, description="发送请求到OpenAI后，等待多久判定为超时")
    max_retry: int = Field(default=2, description="如果OpenAI不响应（网络卡顿、代理失败、KEY失效），重试的次数限制")

    azure_endpoint: str = Field(
        default="https://你亲手写的api名称.openai.azure.com/", description="Azure OpenAI API的Endpoint"
    )
    azure_api_key: str = Field(default="填入azure openai api的密钥", description="Azure OpenAI API的密钥")
    azure_engine: str = Field(default="填入你亲手写的部署名", description="Azure OpenAI API的部署名")
    azure_cfg_array: AzureCFGArray = Field(default_factory=AzureCFGArray, description="Azure 多个模型部署+动态切换配置")


class AliyunConfig(BaseModel):
    enable_audio: bool = Field(default=False, description="是否启用阿里云实时语音识别")
    aliyun_token: str = Field(default="", description="阿里云实时语音识别的token")
    aliyun_appkey: str = Field(default="", description="阿里云实时语音识别的appkey")
    aliyun_accesskey: str = Field(default="", description="阿里云实时语音识别的accesskey")
    aliyun_secret: str = Field(default="", description="阿里云实时语音识别的secret")
    tts_type: str = Field(
        default="EDGE_TTS", description="文本转语音服务类型，可选 ['EDGE_TTS', 'LOCAL_SOVITS_API', 'DISABLE']"
    )
    gpt_sovits_url: str = Field(default="", description="GPT-SOVITS 文本转语音服务的运行地址")
    edge_tts_voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="Edge TTS语音")
    dashscope_api_key: str = Field(default="", description="Dashscope API KEY")
    qwen_local_model_selection: str = Field(default="Qwen/Qwen-1_8B-Chat-Int8", description="本地Qwen模型选择")


class BaiduCloudConfig(BaseModel):
    baidu_cloud_api_key: str = Field(default="", description="百度千帆大模型的API_KEY")
    baidu_cloud_secret_key: str = Field(default="", description="百度千帆大模型的SECRET_KEY")
    baidu_cloud_qianfan_model: str = Field(default="ERNIE-Bot", description="百度千帆大模型的具体模型名称")


class XfyunConfig(BaseModel):
    xfyun_appid: str = Field(default="00000000", description="讯飞星火大模型的APPID")
    xfyun_api_secret: str = Field(default="", description="讯飞星火大模型的API_SECRET")
    xfyun_api_key: str = Field(default="", description="讯飞星火大模型的API_KEY")


class ZhipuaiConfig(BaseModel):
    zhipuai_api_key: str = Field(default="", description="智谱AI大模型的API_KEY")


class AnthropicConfig(BaseModel):
    anthropic_api_key: str = Field(default="", description="Anthropic (Claude) API KEY")


class MoonshotConfig(BaseModel):
    moonshot_api_key: str = Field(default="", description="月之暗面 API KEY")


class YiModelConfig(BaseModel):
    yimodel_api_key: str = Field(default="", description="零一万物(Yi Model) API KEY")


class DeepSeekConfig(BaseModel):
    deepseek_api_key: str = Field(default="", description="深度求索(DeepSeek) API KEY")


class TaichuConfig(BaseModel):
    taichu_api_key: str = Field(default="", description="紫东太初大模型 API KEY")


class MathpixConfig(BaseModel):
    mathpix_appid: str = Field(default="", description="Mathpix APPID，用于PDF的OCR功能")
    mathpix_appkey: str = Field(default="", description="Mathpix APPKEY，用于PDF的OCR功能")


class Doc2XConfig(BaseModel):
    doc2x_api_key: str = Field(default="", description="DOC2X的PDF解析服务API KEY")


class CustomAPIKeyPatternConfig(BaseModel):
    custom_api_key_pattern: str = Field(default="", description="自定义API KEY格式")


class GeminiConfig(BaseModel):
    gemini_api_key: str = Field(default="", description="Google Gemini API-Key")


class HuggingFaceConfig(BaseModel):
    huggingface_access_token: str = Field(default="", description="HuggingFace访问令牌，下载LLAMA时起作用")
    daas_server_url: HttpUrl = Field(default="https://api.huggingface.co/", description="HuggingFace DaaS服务器地址")


class GrobidConfig(BaseModel):
    grobid_urls: List[AnyHttpUrl] = Field(
        default=[
            "https://qingxu98-grobid.hf.space",
            "https://qingxu98-grobid2.hf.space",
        ],
        description="GROBID服务器地址（填写多个可以均衡负载），用于高质量地读取PDF文档",
    )


class SearxngConfig(BaseModel):
    searxng_url: str = Field(default="https://cloud-1.agent-matrix.com/", description="Searxng互联网检索服务URL")


class GeneralConfig(BaseModel):
    llm_model: str = Field(
        default="qwen-plus", description="默认使用的语言模型，必须被包含在 `avail_llm_models` 列表中"
    )
    avail_llm_models: List[str] = Field(
        default=[
            "gpt-4-1106-preview",
            "gpt-4-turbo-preview",
            "gpt-4-vision-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo",
            "azure-gpt-3.5",
            "gpt-4",
            "gpt-4-32k",
            "azure-gpt-4",
            "glm-4",
            "glm-4v",
            "glm-3-turbo",
            "gemini-1.5-pro",
            "chatglm3",
            "glm-4-0520",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-flash",
            "qianfan",
            "deepseekcoder",
            "spark",
            "sparkv2",
            "sparkv3",
            "sparkv3.5",
            "sparkv4",
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-local",
            "qwen-math-plus",
            "qwen-math-turbo",
            "moonshot-v1-128k",
            "moonshot-v1-32k",
            "moonshot-v1-8k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-0125",
            "gpt-4o-2024-05-13",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-2.1",
            "claude-instant-1.2",
            "moss",
            "llama2",
            "chatglm_onnx",
            "internlm",
            "jittorllms_pangualpha",
            "jittorllms_llama",
            "deepseek-chat",
            "deepseek-coder",
            "gemini-1.5-flash",
            "yi-34b-chat-0205",
            "yi-34b-chat-200k",
            "yi-large",
            "yi-medium",
            "yi-spark",
            "yi-large-turbo",
            "yi-large-preview",
        ],
        description="可用的语言模型列表",
    )
    multi_query_llm_models: str = Field(
        default="qwen-max&qwen-math",
        description="定义界面上“询问多个GPT模型”插件应该使用哪些模型，请从 `avail_llm_models` 中选择，并在不同模型之间用`&`间隔",
    )
    default_fn_groups: List[str] = Field(default=["对话", "编程", "学术", "智能体"], description="插件分类默认选项")
    theme: str = Field(
        default="Default",
        description="色彩主题",
    )
    available_themes: List[str] = Field(
        default=["Default", "Chuanhu-Small-and-Beautiful", "High-Contrast", "Gstaff/Xkcd", "NoCrypt/Miku"],
        description="可用的主题列表",
    )
    init_sys_prompt: str = Field(
        default="Serve me as a writing and programming assistant.", description="默认的系统提示词"
    )
    chatbot_height: int = Field(default=1115, description="对话窗的高度（仅在LAYOUT='TOP-DOWN'时生效）")
    code_highlight: bool = Field(default=True, description="是否启用代码高亮")
    layout: str = Field(default="LEFT-RIGHT", description="窗口布局，可选 ['LEFT-RIGHT', 'TOP-DOWN']")
    dark_mode: bool = Field(default=True, description="是否启用暗色模式")
    web_port: int = Field(default=8999, description="网页的端口, -1代表随机端口")
    auto_open_browser: bool = Field(default=False, description="是否自动打开浏览器页面")
    concurrent_count: int = Field(default=100, description="设置gradio的并行线程数")
    auto_clear_txt: bool = Field(default=False, description="是否在提交时自动清空输入框")
    add_waifu: bool = Field(default=False, description="是否加一个live2d装饰")
    authentication: List[List[str]] = Field(default=[], description="设置用户名和密码")
    custom_path: str = Field(default="/", description="如果需要在二级路径下运行")
    ssl_keyfile: str = Field(default="", description="HTTPS秘钥文件路径")
    ssl_certfile: str = Field(default="", description="HTTPS证书文件路径")
    api_org: str = Field(default="", description="极少数情况下，openai的官方KEY需要伴随组织编码使用")
    slack_claude_bot_id: str = Field(default="", description="如果需要使用Slack Claude Bot ID")
    slack_claude_user_token: str = Field(default="", description="如果需要使用Slack Claude User Token")
    chatglm_ptuning_checkpoint: str = Field(default="", description="如果使用ChatGLM2微调模型，请指定模型路径")
    local_model_device: str = Field(default="cpu", description="本地LLM模型如ChatGLM的执行方式，可选 ['cpu', 'cuda']")
    local_model_quant: str = Field(default="FP16", description="本地模型量化，可选 ['FP16', 'INT4', 'INT8']")
    allow_reset_config: bool = Field(
        default=False, description="是否允许通过自然语言描述修改本页的配置，该功能具有一定的危险性，默认关闭"
    )
    autogen_use_docker: bool = Field(default=False, description="在使用AutoGen插件时，是否使用Docker容器运行代码")
    path_private_upload: str = Field(default="private_upload", description="临时的上传文件夹位置")
    path_logging: str = Field(default="gpt_log", description="日志文件夹的位置")
    arxiv_cache_dir: str = Field(default="gpt_log/arxiv_cache", description="存储翻译好的arxiv论文的路径")
    when_to_use_proxy: List[str] = Field(
        default=[
            "Download_LLM",
            "Download_Gradio_Theme",
        ],
        description="除了连接OpenAI之外，还有哪些场合允许使用代理",
    )
    plugin_hot_reload: bool = Field(default=False, description="启用插件热加载")
    num_custom_basic_btn: int = Field(default=4, description="自定义按钮的最大数量限制")


class Configuration(BaseModel):
    openai_config: OpenAIConfig = Field(default_factory=OpenAIConfig, description="OpenAI 配置")
    aliyun_config: AliyunConfig = Field(default_factory=AliyunConfig, description="阿里云 配置")
    baidu_cloud_config: BaiduCloudConfig = Field(default_factory=BaiduCloudConfig, description="百度云 配置")
    xfyun_config: XfyunConfig = Field(default_factory=XfyunConfig, description="讯飞星火 配置")
    zhipuai_config: ZhipuaiConfig = Field(default_factory=ZhipuaiConfig, description="智谱AI 配置")
    anthropic_config: AnthropicConfig = Field(default_factory=AnthropicConfig, description="Anthropic 配置")
    moonshot_config: MoonshotConfig = Field(default_factory=MoonshotConfig, description="月之暗面 配置")
    yi_model_config: YiModelConfig = Field(default_factory=YiModelConfig, description="零一万物 配置")
    deepseek_config: DeepSeekConfig = Field(default_factory=DeepSeekConfig, description="深度求索 配置")
    taichu_config: TaichuConfig = Field(default_factory=TaichuConfig, description="紫东太初 配置")
    mathpix_config: MathpixConfig = Field(default_factory=MathpixConfig, description="Mathpix 配置")
    doc2x_config: Doc2XConfig = Field(default_factory=Doc2XConfig, description="DOC2X 配置")
    custom_api_key_pattern_config: CustomAPIKeyPatternConfig = Field(
        default_factory=CustomAPIKeyPatternConfig, description="自定义API KEY格式 配置"
    )
    gemini_config: GeminiConfig = Field(default_factory=GeminiConfig, description="Google Gemini 配置")
    huggingface_config: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig, description="HuggingFace 配置")
    grobid_config: GrobidConfig = Field(default_factory=GrobidConfig, description="GROBID 配置")
    searxng_config: SearxngConfig = Field(default_factory=SearxngConfig, description="Searxng 配置")
    general_config: GeneralConfig = Field(default_factory=GeneralConfig, description="通用配置")


if __name__ == "__main__":

    pass
