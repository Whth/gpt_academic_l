from os import PathLike
from pathlib import Path

import toml
from toml import dumps

from config_model import Configuration  # 假设Configuration类在config_model模块中


def load_config(config_file: PathLike | str = "./config/config.toml") -> Configuration:
    """
    Load the configuration from a TOML file.
    """
    config_file = Path(config_file)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            config_data = toml.load(f)
        return Configuration(**config_data)
    else:
        config_file.parent.mkdir(parents=True)
        conf = Configuration()
        with open(config_file, "w", encoding="utf-8") as fp:
            fp.write(dumps(conf.model_dump()))
        return conf


config = load_config()

# 将配置项绑定到全局变量
API_KEY = config.openai_config.api_key
USE_PROXY = config.openai_config.use_proxy
proxies = config.openai_config.proxies
EMBEDDING_MODEL = config.openai_config.embedding_model
API_URL_REDIRECT = config.openai_config.api_url_redirect
DEFAULT_WORKER_NUM = config.openai_config.default_worker_num
TIMEOUT_SECONDS = config.openai_config.timeout_seconds
MAX_RETRY = config.openai_config.max_retry
AZURE_ENDPOINT = config.openai_config.azure_endpoint
AZURE_API_KEY = config.openai_config.azure_api_key
AZURE_ENGINE = config.openai_config.azure_engine
AZURE_CFG_ARRAY = config.openai_config.azure_cfg_array.azure_models

MULTI_QUERY_LLM_MODELS = config.general_config.multi_query_llm_models
LLM_MODEL = config.general_config.llm_model
AVAIL_LLM_MODELS = config.general_config.avail_llm_models
THEME = config.general_config.theme
AVAIL_THEMES = config.general_config.available_themes
INIT_SYS_PROMPT = config.general_config.init_sys_prompt
CHATBOT_HEIGHT = config.general_config.chatbot_height
CODE_HIGHLIGHT = config.general_config.code_highlight
LAYOUT = config.general_config.layout
DARK_MODE = config.general_config.dark_mode
WEB_PORT = config.general_config.web_port
AUTO_OPEN_BROWSER = config.general_config.auto_open_browser
CONCURRENT_COUNT = config.general_config.concurrent_count
AUTO_CLEAR_TXT = config.general_config.auto_clear_txt
ADD_WAIFU = config.general_config.add_waifu
AUTHENTICATION = config.general_config.authentication
CUSTOM_PATH = config.general_config.custom_path
SSL_KEYFILE = config.general_config.ssl_keyfile
SSL_CERTFILE = config.general_config.ssl_certfile
API_ORG = config.general_config.api_org
SLACK_CLAUDE_BOT_ID = config.general_config.slack_claude_bot_id
SLACK_CLAUDE_USER_TOKEN = config.general_config.slack_claude_user_token
ALLOW_RESET_CONFIG = config.general_config.allow_reset_config
AUTOGEN_USE_DOCKER = config.general_config.autogen_use_docker
PATH_PRIVATE_UPLOAD = config.general_config.path_private_upload
PATH_LOGGING = config.general_config.path_logging
ARXIV_CACHE_DIR = config.general_config.arxiv_cache_dir
WHEN_TO_USE_PROXY = config.general_config.when_to_use_proxy
PLUGIN_HOT_RELOAD = config.general_config.plugin_hot_reload
NUM_CUSTOM_BASIC_BTN = config.general_config.num_custom_basic_btn
DEFAULT_FN_GROUPS = config.general_config.default_fn_groups
CHATGLM_PTUNING_CHECKPOINT = config.general_config.chatglm_ptuning_checkpoint
# 本地LLM模型如ChatGLM的执行方式 CPU/GPU
LOCAL_MODEL_DEVICE = config.general_config.local_model_device
LOCAL_MODEL_QUANT = config.general_config.local_model_quant

QWEN_LOCAL_MODEL_SELECTION = config.aliyun_config.qwen_local_model_selection
DASHSCOPE_API_KEY = config.aliyun_config.dashscope_api_key
ENABLE_AUDIO = config.aliyun_config.enable_audio
ALIYUN_TOKEN = config.aliyun_config.aliyun_token
ALIYUN_APPKEY = config.aliyun_config.aliyun_appkey
ALIYUN_ACCESSKEY = config.aliyun_config.aliyun_accesskey
ALIYUN_SECRET = config.aliyun_config.aliyun_secret
TTS_TYPE = config.aliyun_config.tts_type
GPT_SOVITS_URL = config.aliyun_config.gpt_sovits_url
EDGE_TTS_VOICE = config.aliyun_config.edge_tts_voice

BAIDU_CLOUD_API_KEY = config.baidu_cloud_config.baidu_cloud_api_key
BAIDU_CLOUD_SECRET_KEY = config.baidu_cloud_config.baidu_cloud_secret_key
BAIDU_CLOUD_QIANFAN_MODEL = config.baidu_cloud_config.baidu_cloud_qianfan_model

XFYUN_APPID = config.xfyun_config.xfyun_appid
XFYUN_API_SECRET = config.xfyun_config.xfyun_api_secret
XFYUN_API_KEY = config.xfyun_config.xfyun_api_key

ZHIPUAI_API_KEY = config.zhipuai_config.zhipuai_api_key
ANTHROPIC_API_KEY = config.anthropic_config.anthropic_api_key
MOONSHOT_API_KEY = config.moonshot_config.moonshot_api_key
YIMODEL_API_KEY = config.yi_model_config.yimodel_api_key
DEEPSEEK_API_KEY = config.deepseek_config.deepseek_api_key
TAICHU_API_KEY = config.taichu_config.taichu_api_key

MATHPIX_APPID = config.mathpix_config.mathpix_appid
MATHPIX_APPKEY = config.mathpix_config.mathpix_appkey


CUSTOM_API_KEY_PATTERN = config.custom_api_key_pattern_config.custom_api_key_pattern

GEMINI_API_KEY = config.gemini_config.gemini_api_key

HUGGINGFACE_ACCESS_TOKEN = config.huggingface_config.huggingface_access_token
DAAS_SERVER_URL = config.huggingface_config.daas_server_url
GROBID_URLS = config.grobid_config.grobid_urls
SEARXNG_URL = config.searxng_config.searxng_url
