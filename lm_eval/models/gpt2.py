import torch
from queue import Queue
import threading
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM

queue1 = Queue()

def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            # Initialize new model and tokenizer instances
            cudaID0 = self.device
            print(f"gpt2.py::transformers.AutoModelForCausalLM.from_pretrained::{cudaID0}")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                cudaID = 0,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(self.device)
            print(f"Model0:{self.model}")
       
            cudaID1 = 1
            cudaID1_device = 'cuda:1'
            print(f"gpt2.py::transformers.AutoModelForCausalLM.from_pretrained::{cudaID1_device}")
            self.model1 = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                cudaID = cudaID1,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(cudaID1_device)
            print(f"Model1:{self.model1}")
            
            cudaID2 = 2
            cudaID2_device = 'cuda:2'
            print(f"gpt2.py::transformers.AutoModelForCausalLM.from_pretrained::{cudaID2_device}")
            self.model2 = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                cudaID = cudaID2,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(cudaID2_device)
            print(f"Model2:{self.model2}")
            
            cudaID3 = 3
            cudaID3_device = 'cuda:3'
            print(f"gpt2.py::transformers.AutoModelForCausalLM.from_pretrained::{cudaID3_device}")
            self.model3 = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                cudaID = cudaID3,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(cudaID3_device)
            print(f"Model3:{self.model3}")
            
            print("gpt2.py::transformers.AutoTokenizer.from_pretrained")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else pretrained,
                cudaID = 0,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        else:
            raise TypeError(
                "Parameter pretrained should be of type str or transformers.PreTrainedModel"
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            # Pre-processing Model (Embedding)
            hidden_states, att_mask = self.model(inps)
            while True:
                if queue1.empty() == True:
                    queue1.put((hidden_states, att_mask))
                    break
                else: continue

    def _decode_func(self, hidden_states, att_mask):
        # Decoding Model (l0 ~ l3)
        hidden_states = hidden_states.to('cuda:1')
        att_mask = att_mask.to('cuda:1')
        hidden_states1, att_mask1 = self.model1(hidden_states, att_mask)
        
        # Decoding Model (l4 ~ l7)
        hidden_states1 = hidden_states1.to('cuda:2')
        att_mask1 = att_mask1.to('cuda:2')
        hidden_states2, att_mask2 = self.model2(hidden_states1, att_mask1)
        
        # Decoding (l8 ~ l11) + Post-processing Model
        hidden_states2 = hidden_states2.to('cuda:3')
        att_mask2 = att_mask2.to('cuda:3')
        output = self.model3(hidden_states2, att_mask2)[0]
        return output
    
    def _trigger(self, id):
        if id == 1:
            data = queue1.get()
            hidden_states = data[0]
            att_mask = data[1]
            return self._decode_func(hidden_states, att_mask)
    
    def _buffer_empty(self, idx):
        if idx == 1:
            if not queue1.empty():
                return False
            return True
    
    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM
