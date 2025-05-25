from fastapi.middleware.cors import CORSMiddleware
from kserve import logging, model_server
from kserve.model import PredictorConfig
from kserve.model_server import ModelServer, app
from vllm import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

from .llm import QwenLLM

DEFAULT_MODEL_PATH = '/mnt/models'


def main_llm():
    # FlexibleArgumentParser(ArgumentParser)
    parser = FlexibleArgumentParser(parents=[model_server.parser])
    AsyncEngineArgs.add_cli_args(parser)
    args, _ = parser.parse_known_args()

    if args.configure_logging:
        logging.configure_logging(args.log_config_file)

    predictor_config = PredictorConfig(
      predictor_host=args.predictor_host,
      predictor_protocol=args.predictor_protocol,
      predictor_use_ssl=args.predictor_use_ssl,
    )

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.model = engine_args.tokenizer = DEFAULT_MODEL_PATH

    model = QwenLLM(args.model_name, engine_args=engine_args, predictor_config=predictor_config)
    model.load()
    app.add_middleware(
      CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*']
    )
    ModelServer().start([model])


if __name__ == '__main__':
    main_llm()
