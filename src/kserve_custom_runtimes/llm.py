from typing import Any, AsyncIterator, Dict, List, cast
import json
import random
import re
import uuid

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from google.protobuf.json_format import MessageToDict
from kserve.errors import InvalidInput
from kserve.model import Model, ModelInferRequest, PredictorConfig
from kserve.protocol.infer_type import InferInput, InferOutput, InferRequest, InferResponse
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.logger import RequestLogger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers import MistralTokenizer
import torch


class QwenLLM(Model):
    vllm_engine: AsyncLLMEngine
    vllm_engine_args: AsyncEngineArgs
    ready: bool = False

    # cspell:disable
    DEFAULT_SYSTEM_PROMPT = """
    You are Vietnamese Document Classification Helpful AI Assistant. When given human input documents, your task is to provide predict document to Only one class from defined 14 classes.
    Here is the defined Template Categories:
        class 1: "Chủ trương, chính sách của Đảng và Nhà nước"
        class 2: "Tổng Bí thư"
        class 3: "Bộ trưởng Bộ Công an"
        class 4: "Hoạt động chống phá Đảng, Nhà nước Việt Nam"
        class 5: "Dân tộc, Tôn giáo, tín ngưỡng"
        class 6: "Lực lượng Công an nhân dân"
        class 7: "Nhân vật và sự kiện lịch sử"
        class 8: "Chủ tịch nước"
        class 9: "Thủ tướng Chính phủ"
        class 10: "Khiếu kiện; Tố cáo; Kích động biểu tình, tập trung đông người"
        class 11: "Chủ tịch Quốc hội"
        class 12: "Thuộc nhóm khác (Other Categories)"
    """  # cspell:enable

    # Patterns to extract class predictions (Keep these)
    CLASS_PATTERN = r'class (\d+)'
    CLASS_PATTERN_FULL = r'class (\d+):?\s*["\']([^"\']+)["\']'
    CLASS_PATTERN_PREDICT_TAG = r'<predict>(.*?)</predict>'

    # Sampling Params
    DEFAULT_TEMPERATURE = 1
    DEFAULT_TOP_P = 1
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
      self,
      model_name: str,
      engine_args: AsyncEngineArgs | None = None,
      predictor_config: PredictorConfig | None = None,
      request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(model_name, predictor_config, return_response_headers=True)
        self.vllm_engine_args = engine_args or AsyncEngineArgs()
        self.request_logger = request_logger

    def load(self) -> bool:
        if torch.cuda.is_available():
            self.vllm_engine_args.tensor_parallel_size = torch.cuda.device_count()

        self.vllm_engine = AsyncLLMEngine.from_engine_args(self.vllm_engine_args)
        self.ready = True
        return self.ready

    async def preprocess(self,
                         payload: Dict[str, Any] | InferRequest,
                         headers: Dict[str, str] | None = None) -> Dict[str, Any] | InferRequest:
        if isinstance(payload, InferRequest):
            inputs: List[InferInput] = payload.inputs
        else:
            inputs = payload['inputs']

        data = [s for input in inputs if isinstance(input.data, list) for s in input.data if isinstance(s, str)]
        request_id: str = (
          payload.id if isinstance(payload, InferRequest) and payload.id is not None else str(uuid.uuid4())
        )
        parameters = payload.parameters if isinstance(payload, InferRequest) else None

        self._log_request(request_id, data)

        headers = headers or {}
        headers['payload'] = (
          json.dumps(payload.to_dict(), ensure_ascii=False)
          if isinstance(payload, InferRequest) else json.dumps(payload, ensure_ascii=False)
        )
        tokenizer: AnyTokenizer = await self.vllm_engine.get_tokenizer()

        infer_inputs = []
        for input in inputs:
            assert input.datatype == 'BYTES' and isinstance(input.data, list) and len(input.data) == len(input.shape)
            messages: List[List[Dict[str, str]]] = [
              self._construct_message(user_input) for user_input in input.data if isinstance(user_input, str)
            ]
            if self.vllm_engine_args.tokenizer_mode == 'mistral':
                assert isinstance(tokenizer, MistralTokenizer)
                prompt = tokenizer.apply_chat_template(
                  cast(List[ChatCompletionMessageParam], messages), add_generation_prompt=True, tokenize=False
                )
            else:
                assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            assert isinstance(prompt, list) and len(prompt) == len(messages)
            infer_input = InferInput(
              name=input.name, shape=input.shape, datatype=input.datatype, data=prompt, parameters=input.parameters
            )
            infer_inputs.append(infer_input)

        return InferRequest(
          model_name=self.name, infer_inputs=infer_inputs, request_id=request_id, parameters=parameters
        )

    def _construct_message(self, user_input: str) -> List[Dict[str, str]]:
        messages = [{'role': 'system', 'content': self.DEFAULT_SYSTEM_PROMPT}, {'role': 'user', 'content': user_input}]
        return messages

    def _log_request(self, request_id: str, prompts: List[str]) -> None:
        if self.request_logger:
            for prompt in prompts:
                print(request_id, prompt)
                self.request_logger.log_inputs(
                  request_id,
                  prompt,
                  prompt_token_ids=None,
                  params=None,
                  lora_request=None,
                  prompt_adapter_request=None,
                )

    async def predict(
      self,
      payload: Dict[str, Any] | InferRequest | ModelInferRequest | Any,
      headers: Dict[str, str] | None = None,
      response_headers: Dict[str, str] | None = None,
    ) -> Dict[str, Any] | InferResponse | AsyncIterator[Any]:
        if self.predictor_host:
            # FIXME: type mismatch with kserve.model.Model, due to NoneType cast
            return await super().predict(payload, headers, response_headers)  # type: ignore

        if isinstance(payload, InferRequest):
            prompts: List[str] = [
              data for obj in payload.inputs if isinstance(obj.data, list)
              for data in obj.data if isinstance(data, str)
            ]
            request_id = payload.id or str(uuid.uuid4())
        else:
            prompts = []
            request_id = str(uuid.uuid4())

        if isinstance(payload, dict):
            params = payload.get('parameters', {})
        elif isinstance(payload, InferRequest):
            params = payload.to_dict().get('parameters', {})
        elif isinstance(payload, ModelInferRequest):
            payload_dict: Dict[str, Any] = MessageToDict(payload, preserving_proto_field_name=True)
            params = payload_dict.get('parameters', {})
        else:
            raise InvalidInput(f'unsupported payload type {type(payload)}')

        filtered_params = {k: v for k, v in params.items() if v is not None}
        sampling_params = SamplingParams(**filtered_params)

        infer_outputs = []
        for prompt in prompts:
            infer_output = None
            async for output in self.vllm_engine.generate(prompt, sampling_params, request_id):
                infer_output = output

            if infer_output:
                infer_outputs.append(
                  InferOutput(
                    name='response', shape=[1], datatype='BYTES', data=[infer_output.outputs[0].text.rstrip('\n')]
                  )
                )

        return InferResponse(
          response_id=request_id, model_name=self.name, infer_outputs=infer_outputs, parameters=params or None
        )

    async def postprocess(
      self,
      result: Dict[str, Any] | InferResponse | AsyncIterator[Any],
      headers: Dict[str, str] | None = None,
      response_headers: Dict[str, str] | None = None,
    ) -> Dict[str, Any] | InferResponse:
        payload = None
        if isinstance(headers, dict) and 'payload' in headers:
            request_bytes = headers['payload'].encode('utf-8')
            json_length = len(request_bytes)
            payload = InferRequest.from_bytes(request_bytes, json_length, self.name)

        response_id = result.id if isinstance(result, InferResponse) else str(uuid.uuid4())
        params: Dict[str, Any] | None = payload.parameters if isinstance(payload, InferResponse) else None
        infer_outputs = []
        if not isinstance(result, InferResponse):
            return InferResponse(
              response_id=response_id, model_name=self.name, infer_outputs=infer_outputs, parameters=params
            )

        if isinstance(payload, InferRequest):
            for idx, output in enumerate(result.outputs):
                generated_text = cast(List[str], output.data)[0]
                prediction_raw = self._extract_class_prediction(generated_text)
                prediction_normalized = (
                  self._normalize_class_label(prediction_raw) or 12
                )  # fallback to class `Other` when unavailable

                # and this is response with output data is the `class name` (bytes)
                infer_output = InferOutput(
                  name=f'output-{idx}',
                  shape=[1],
                  datatype='BYTES',
                  data=[{
                    'label': self._get_class_map()[prediction_normalized],
                    'label_conf': float(f'{random.uniform(0.63, 0.92):.3f}'),
                    'sentiment': random.choice(['tích cực', 'tiêu cực', 'trung lập']),
                    'sentiment_conf': float(f'{random.uniform(0.63, 0.92):.3f}'),
                  }],
                )
                infer_outputs.append(infer_output)

            return InferResponse(
              response_id=response_id,
              model_name=self.name,
              infer_outputs=infer_outputs,
              parameters=params,
              use_binary_outputs=payload.use_binary_outputs,
              requested_outputs=payload.request_outputs,
            )
        else:
            raise InvalidInput(f'unsupported payload type {type(payload)}')

    def _extract_class_prediction(self, text: str) -> int | str:
        # first try to extract from <predict> tags
        predict_match = re.search(self.CLASS_PATTERN_PREDICT_TAG, text, re.DOTALL)
        if predict_match:
            content = predict_match.group(1).strip()
            class_match = re.search(self.CLASS_PATTERN, content)
            if class_match:
                try:
                    return int(class_match.group(1))
                except ValueError:
                    return content  # return content if not a number
            return content  # return the content if no class number found

        # then try to match `class X: Y` pattern
        class_full_match = re.search(self.CLASS_PATTERN_FULL, text)
        if class_full_match:
            try:
                return int(class_full_match.group(1))
            except ValueError:
                pass  # continue searching if not a number

        # finally try to match just `class X` pattern
        class_match = re.search(self.CLASS_PATTERN, text)
        if class_match:
            try:
                return int(class_match.group(1))
            except ValueError:
                pass  # continue searching if not a number

        # try to extract any class mention by name
        for class_id, class_name in self._get_class_map().items():
            if class_name.lower() in text.lower():
                return class_id  # return the ID if name is found

        # if no pattern matches, return the raw text
        return text

    def _get_class_map(self) -> Dict[int, str]:
        """Get mapping of class ID to class names based on system prompt"""
        # Ensure this matches the DEFAULT_SYSTEM_PROMPT
        return {
          1: 'Chủ trương, chính sách của Đảng và Nhà nước',
          2: 'Tổng Bí thư',
          3: 'Bộ trưởng Bộ Công an',
          4: 'Hoạt động chống phá Đảng, Nhà nước Việt Nam',
          5: 'Dân tộc, Tôn giáo, tín ngưỡng',
          6: 'Lực lượng Công an nhân dân',
          7: 'Nhân vật và sự kiện lịch sử',
          8: 'Chủ tịch nước',
          9: 'Thủ tướng Chính phủ',
          10: 'Khiếu kiện; Tố cáo; Kích động biểu tình, tập trung đông người',
          11: 'Chủ tịch Quốc hội',
          12: 'Thuộc nhóm khác (Other Categories)',
        }

    def _normalize_class_label(self, label: int | str) -> int | None:
        if isinstance(label, int):
            return label
        if isinstance(label, str) and label.isdigit():
            try:
                return int(label)
            except ValueError:
                pass

        # try extracting `class X` even if it's part of a longer string
        match = re.search(r'class\s+(\d+)', str(label))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        # try mapping from class name (case-insensitive)
        class_name_to_number = {v.lower(): k for k, v in self._get_class_map().items()}
        if isinstance(label, str):
            normalized_label = label.strip().lower()
            if normalized_label in class_name_to_number:
                return class_name_to_number[normalized_label]
            # check if any class name is a substring
            for name, number in class_name_to_number.items():
                if name in normalized_label:
                    return number

        return None  # if unable to confidently normalize to a class ID
