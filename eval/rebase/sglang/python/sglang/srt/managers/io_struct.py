import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from sglang.srt.sampling_params import SamplingParams


@dataclass
class GenerateReqInput:
    # The input prompt
    text: Union[List[str], str]
    # The image input
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params
    sampling_params: Union[List[Dict], Dict] = None
    # The request id
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs
    return_logprob: Optional[Union[List[bool], bool]] = None
    # The start location of the prompt for return_logprob
    logprob_start_len: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in logprobs
    return_text_in_logprobs: bool = False
    # Whether to stream output
    stream: bool = False
    # Whether only do forward
    forward_only: bool = False
    # Whether truncate to force last token
    logits_require_id: Optional[int] = None

    def post_init(self):
        is_single = isinstance(self.text, str)

        if is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if "forward_only" in self.sampling_params:
                self.forward_only = self.sampling_params["forward_only"]
            if "logits_require_id" in self.sampling_params:
                self.logits_require_id = self.sampling_params["logits_require_id"]
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = 0
        else:
            num = len(self.text)

            if self.image_data is None:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num
            if "forward_only" in self.sampling_params[0]:
                self.forward_only = self.sampling_params[0]["forward_only"]
            if "logits_require_id" in self.sampling_params[0]:
                self.logits_require_id = self.sampling_params[0]["logits_require_id"]

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                assert isinstance(self.rid, list)

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num

            if self.logprob_start_len is None:
                self.logprob_start_len = [0] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num


@dataclass
class TokenizedGenerateReqInput:
    rid: str
    input_text: str
    input_ids: List[int]
    pixel_values: List[float]
    image_hash: int
    image_size: List[int]
    sampling_params: SamplingParams
    return_logprob: bool
    logprob_start_len: int
    stream: bool
    forward_only: bool
    logits_require_id: Optional[int]


@dataclass
class BatchTokenIDOut:
    rids: List[str]
    output_tokens: List[List[int]]
    output_and_jump_forward_strs: List[str]
    hit_stop_str: List[Optional[str]]
    skip_special_tokens: List[bool]
    meta_info: List[Dict]
    finished: List[bool]
    scores: List[List[float]]
    forward_only: List[bool]


@dataclass
class BatchStrOut:
    rids: List[str]
    output_str: List[str]
    meta_info: List[Dict]
    finished: List[bool]
    scores: List[List[float]]
    forward_only: List[bool]


@dataclass
class FlushCacheReq:
    pass


@dataclass
class DetokenizeReqInput:
    input_ids: List[int]
