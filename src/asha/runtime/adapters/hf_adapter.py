# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import BaseAdapter


class HuggingFaceAdapter(BaseAdapter):

    def __init__(self, model: str | None = None) -> None:
        """Initialize HuggingFace adapter with specified model.

        Args:
            model: HuggingFace model name or path

        Raises:
            ImportError: If transformers library not installed
            ValueError: If model name not provided
            RuntimeError: If model loading fails
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is not installed. "
                "Install with: pip install transformers torch"
            )

        if not model:
            raise ValueError(
                "Model name is required for HuggingFace adapter. "
                "Example: HuggingFaceAdapter(model='microsoft/DialoGPT-medium')"
            )

        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Try using pipeline for easier handling
            self.generator = pipeline(
                "text-generation",
                model=model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=0 if self.device == "cuda" else -1,
                trust_remote_code=True,
            )
        except Exception as e:
            # Fallback to manual model loading
            print(f"Warning: Pipeline failed, trying manual loading: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                    trust_remote_code=True,
                )
                # Move model to device manually
                if self.device == "cuda":
                    self.model = self.model.to("cuda")
                self.generator = None
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to load HuggingFace model: {fallback_error}. "
                    "Check model name, internet connection, and available disk space."
                )

    def generate(self, prompt: str) -> str:
        """Generate response using HuggingFace model.

        Args:
            prompt: Input prompt for HuggingFace model

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
            RuntimeError: If model not properly initialized
        """
        try:
            if self.generator:
                # Use pipeline if available
                outputs = self.generator(
                    prompt,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                    return_full_text=False,
                )
                response = outputs[0]["generated_text"]
            else:
                # Use manual model loading
                inputs = self.tokenizer(prompt, return_tensors="pt").to(
                    self.model.device
                )

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove the original prompt from the response
                if response.startswith(prompt):
                    response = response[len(prompt) :].strip()

            return str(response).strip()

        except Exception as e:
            return f"HuggingFace generation error: {str(e)}"
